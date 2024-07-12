import torch
import traceback

from .code_cache import CodeCache
from .fusion import pointwise_fusion
from .utils import lazy_graph_print_tabular, lazy_module_print_readable

from torch._dynamo import lookup_backend
#from .fused_rms_quant import setup_fused_rms_norm
from .silu_mul_quant import setup_silu_mul_quant
from torch.fx.passes.shape_prop import ShapeProp
from typing import List, Optional, Callable

from vllm.logger import init_logger

logger = init_logger(__name__)

###############################################################################
#
# Backend
#
###############################################################################


def optimize(
    cc: CodeCache,
    mod: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
) -> torch.fx.GraphModule:
    """
    Run optimizer on the given module.  Future optimization passes will
    be called from here.
    """
    mod = pointwise_fusion(cc, mod, example_inputs)
    return mod


def backend_compile(gm: torch.fx.GraphModule,
                    example_inputs: List[torch.Tensor],
                    backend: Optional[str] = 'inductor') -> Callable:
    """
    Compile a module with the given backend.
    """
    if not backend:
        return gm.forward

    def maybe_name(gm: torch.fx.GraphModule) -> Optional[str]:
        return gm.name if hasattr(gm, 'name') else None

    try:
        backend_fn = lookup_backend(backend)
        logger.debug("attempting %s on %s", backend, maybe_name(gm))
        backend_compiled = backend_fn(gm, example_inputs)
        if backend_compiled is not None:
            logger.debug("%s compiled %s.", backend, maybe_name(gm))
            return backend_compiled
    except Exception as ex:
        logger.warning("backend_compile failed: %s", ex)
        logger.warning("Trace: %s", traceback.format_tb(ex.__traceback__))
        pass

    return gm.forward


class backend_class:
    """
    A custom backend for torch.compile.

    This backend works by partitioning the provided module into supported/unsupported
    sub-modules.  The supported submodules are passed to an optimizer and then compiled
    via an optional "final" backend.
    """

    # This is a global code cache that applies to all models.
    # TODO: this might need additional context to avoid collisions, e.g.
    # module/model name.
    cc = CodeCache(disable=False)

    def __init__(self, backend: Optional[str] = 'inductor'):
        self.backend = backend
        # setup_fused_rms_norm(backend_class.cc)
        setup_silu_mul_quant(backend_class.cc)

    def __call__(self, gm: torch.fx.GraphModule,
                 example_inputs: List[torch.Tensor]) -> Callable:
        logger.info("Graph optimizer start")

        logger.debug("Original module:")
        logger.debug(
            lazy_graph_print_tabular(gm.graph, 'users',
                                     lambda n: list(n.users.keys())))
        logger.debug("input_types: %s", [type(inp) for inp in example_inputs])

        # Annotate all nodes with types and shapes.
        ShapeProp(gm).propagate(*example_inputs)

        gm = optimize(backend_class.cc, gm, example_inputs)

        # TODO: no need to recompile if nothing got optimized.
        gm.recompile()

        logger.debug("Final module:")
        logger.debug(lazy_module_print_readable(gm, False))

        # Forward optimized graph onto "final" backend (if any).
        fn = backend_compile(gm, example_inputs, backend=self.backend)

        logger.info("Graph optimizer end")

        return fn


def backend(gm: torch.fx.GraphModule,
            example_inputs: List[torch.Tensor]) -> Callable:
    """
    The default custom backend function for use with torch.compile.
    """
    return backend_class()(gm, example_inputs)


def make_backend(backend: Optional[str] = 'inductor') -> backend_class:
    """
    Construct a custom torch.compile backend with optional 'final' backend for
    optimized subgraphs. The default 'final' backend is the inductor. None can
    be used instead to leave optimized subgraphs as interpreted.
    """
    return backend_class(backend)


# TODO: come up with better name for this
def optimizer(_func: Optional[Callable] = None,
              backend: Optional[str] = None,
              fullgraph: bool = False):

    def body(fn: Callable) -> Callable:
        return torch.compile(fn,
                             backend=make_backend(backend=backend),
                             fullgraph=fullgraph)

    return body if _func is None else body(_func)
