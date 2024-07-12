###############################################################################
#
# Operator fusion pass
#
###############################################################################

import operator
import torch

from .code_cache import CodeCache
from .fused_op_generator import FusionFail
from .naive_fused_op_generator import NaiveFusedOpGenerator
from .register import FUSABLE
from .utils import FlowGraph, node_function_target, graph_print_tabular, SubGraph, is_call, call_method_class, lazy_graph_print_tabular, mangle_name

from torch.fx.passes.shape_prop import ShapeProp
from typing import List, Tuple, Any, Dict, Optional, Callable, Mapping, Set
from vllm.logger import init_logger

logger = init_logger(__name__)


def fuse_graph_nodes(cc: CodeCache, sub: SubGraph):
    """
    Fuse all the nodes in the given sub-graph into a single function call.
    """
    outputs = sub.outputs
    inputs = sub.inputs

    sub.topo_sort()

    # Collect all the nodes that will need to be fused (and erased) later.
    nodes_to_fuse = []
    kwargs: Dict[torch.fx.Node, Dict[str, torch.fx.Argument]] = dict()

    for n in sub.nodes:
        if not is_call(n):
            continue

        nodes_to_fuse.append(n)

    # Lookup or create the fused operation.
    try:
        fn_key = f"{mangle_name(nodes_to_fuse)}_fused"

        def generate() -> Optional[Callable]:
            fgen = NaiveFusedOpGenerator()
            return fgen.make_fused_op(fn_key, inputs, outputs, nodes_to_fuse,
                                      kwargs)

        fn = cc.lookup_or_create(fn_key, generate)

    except FusionFail as ff:
        logger.info(f"fusion failed '{ff}' for subgraph.")
        return

    if fn is None:
        logger.debug(f"fusion failed previously for subgraph.")
        return

    logger.debug(f"fused fn = {fn}")

    #
    # Update the graph
    # 1. insert the call_function for the fused op
    # 2. insert new output node(s)
    # 3. delete old call_function and output nodes.
    #

    insert_point = sub.last_input()
    sub.module.graph.inserting_after(insert_point)

    # Note: we do not update the meta info for cf here.  It should
    # not be required after transformation anyway.
    cf = sub.module.graph.call_function(fn, args=tuple(inputs), kwargs=kwargs)
    logger.debug(f"fused op: {cf.format_node()}, num_outputs={len(outputs)}")

    new_sub = [cf]

    # Note: assumes single output
    if len(outputs) == 1:
        outputs[0].replace_all_uses_with(cf, propagate_meta=True)
    else:
        cf_idx = cf
        for i, output in enumerate(outputs):
            sub.module.graph.inserting_after(cf_idx)
            cf_idx = sub.module.graph.call_function(operator.getitem,
                                                    args=(cf, i))
            output.replace_all_uses_with(cf_idx, propagate_meta=True)
            new_sub.append(cf_idx)

    # Erase all the nodes in the subgraph
    sub.erase()

    # Extra build() not necessary but nice for debugging.
    sub.build(new_sub)


def is_fusable(node: torch.fx.Node) -> bool:
    """
    Determine whether or not node is a fusable operations.
    TODO: Smarter filter for 'getitem'.
    """
    if not is_call(node):
        return False

    op_name = node_function_target(node)
    if node.op == 'call_function':
        return op_name in FUSABLE and not FUSABLE[op_name].is_compute
    else:
        # TODO: check class type
        class_type = call_method_class(node)
        return op_name in FUSABLE and not FUSABLE[op_name].is_compute


def is_compute(node: torch.fx.Node) -> bool:
    """
    Determine whether or not node is a fusable compute operation, e.g. gemm.
    """
    if not is_call(node):
        return False

    op_name = node_function_target(node)
    if node.op == 'call_function':
        return op_name in FUSABLE and FUSABLE[op_name].is_compute
    else:
        # TODO: check class type
        class_type = call_method_class(node)
        return op_name in FUSABLE and FUSABLE[op_name].is_compute


def is_fusable_pair(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    """
    Are nodes a and b fusable together?
    This function assumes 'b' is a direct successor of 'a'.
    """
    return is_fusable(a) and is_fusable(b)


def is_compute_fusable_pair(a: torch.fx.Node, b: torch.fx.Node) -> bool:
    """
    Are nodes 'a' and 'b' fusable together and is 'a' optionally a compute op?
    This function assumes 'b' is a direct successor of 'a'.
    """
    return (is_fusable(a) or is_compute(a)) and is_fusable(b)


def supported_kwargs(node: torch.fx.Node,
                     only_const_kwargs: bool = True) -> bool:
    """
    Determine if any kwargs associated with 'node' are supported.
    """
    if only_const_kwargs:
        for arg in node.kwargs.values():
            if not isinstance(arg, torch.fx.node.BaseArgumentTypes):
                return False
        return True
    else:
        return node.kwargs is None or len(node.kwargs) == 0


def dump_partitions(node_map: Dict[torch.fx.Node, int]) -> str:
    parts = dict()
    for n, p in node_map.items():
        if not p in parts:
            parts[p] = set([n])
        else:
            parts[p].add(n)

    part_str = "{\n"
    for p, ns in parts.items():
        part_str = part_str + f"  {p}: {str(ns)}\n"
    part_str = part_str + "\n}"
    return part_str


def non_trivial_op(n: torch.fx.Node) -> bool:
    if not is_call(n):
        return False
    trg = node_function_target(n)
    return not FUSABLE[trg].is_trivial if trg in FUSABLE else False


def is_get_attr(a: torch.fx.Node) -> bool:
    return a.op == 'get_attr'


def pointwise_fusion(cc: CodeCache,
                     mod: torch.fx.GraphModule,
                     example_inputs: List[torch.Tensor],
                     fuse_inputs: bool = False,
                     fuse_with_compute=True) -> torch.fx.GraphModule:
    # find all groups of nodes that can be fused and assign to
    # unique partition id, i.e. node_map

    fg = FlowGraph(mod)

    logger.debug(f"FlowGraph:\n{fg.dump()}")

    ShapeProp(mod).propagate(*example_inputs)

    # Map of nodes to partitions.  Partition 0 represents unfusable operations.
    node_map: Dict[torch.fx.Node, int] = dict()
    partition = 0

    # assumption, graph.nodes are in topo order
    mod.graph.lint()

    logger.debug("Start fusion")

    # create partition groups
    # run in reverse order so predecessors of non-unary ops will appear
    # in the same partition.
    for n in reversed(mod.graph.nodes):
        logger.debug(f"CONSIDER {n}")

        if not is_call(n):
            logger.debug(f"  REJECT {n}: not call")
            node_map[n] = 0
            continue

        pred = is_fusable_pair if not fuse_with_compute else is_compute_fusable_pair

        fusable = [
            pred(s, n) for s in fg.predecessors(n)
            if s.op != 'placeholder' and not is_get_attr(s)
        ]
        if not all(fusable):
            logger.debug(
                f"  REJECT {n}: not all preds fusable: {fusable}, {fg.predecessors(n)}"
            )
            if not n in node_map:
                node_map[n] = 0
            continue

        # don't support anything with kwargs for now
        if not supported_kwargs(n):
            logger.debug(f"  REJECT {n}: unsupported kwargs")
            node_map[n] = 0
            continue

        if n not in node_map:
            partition = partition + 1
            node_map[n] = partition

        for s in fg.predecessors(n):
            if s.op != 'placeholder':
                node_map[s] = node_map[n]

        logger.debug(f"ACCEPT {n}: partition {node_map[n]}")

    logger.debug(f"partitions = {dump_partitions(node_map)}")

    def same_partition(nodes: Set[torch.fx.Node]) -> bool:
        if len(nodes) > 0:
            part = node_map[next(iter(nodes))]
            return all([node_map[n] == part for n in nodes])
        return False

    def only_pointwise(partition: int) -> bool:
        nodes = [n for n, p in node_map.items() if p == partition]
        return all([is_fusable(n) and not is_compute(n) for n in nodes])

    #
    # Fuse pointwise op stacks with compute nodes. The fuse_inputs flag controls
    # prologue or epilogue fusion.
    #
    if fuse_with_compute:
        num_compute: Dict[int, int] = dict()  # use set

        for i in range(len(node_map)):
            num_compute[i] = 0

        for n in mod.graph.nodes:
            part = node_map[n]
            if is_compute(n):
                assert part == 0 or num_compute[
                    part] == 0, f"part = {part}: {[n for n, p in node_map.items() if p == part]}"
                num_compute[part] = num_compute[part] + 1

        for n in mod.graph.nodes:
            if not is_call(n):
                continue

            logger.debug(f"COMPUTE CONSIDER {n}")

            if fuse_inputs:
                nodes = fg.predecessors(n)
            else:
                nodes = fg.successors(n)

            if not is_compute(n):
                logger.debug(f"COMPUTE REJECT {n}: not compute")
                continue

            if not same_partition(nodes):
                logger.debug(
                    f"COMPUTE REJECT {n}: not all neighbors in same partition {str(nodes)}"
                )
                continue

            fuse_part = next(iter(nodes))

            if not only_pointwise(fuse_part):
                logger.debug(
                    f"COMPUTE REJECT {n}: not only_pointwise in users {str(fuse_part)}"
                )
                continue

            fuse_part_id = node_map[fuse_part]

            if fuse_part_id != 0 and fuse_part_id in num_compute and num_compute[
                    fuse_part_id] >= 1:
                logger.debug(
                    f"COMPUTE REJECT {n}: already a compute node in {str(node_map[fuse_part])}"
                )
                continue

            logger.debug(
                f"COMPUTE ACCEPT {n}: partition {fuse_part_id}, nodes={str(nodes)}"
            )

            num_compute[fuse_part_id] = num_compute[fuse_part_id] + 1
            node_map[n] = fuse_part_id

    logger.debug(f"final paritions = {dump_partitions(node_map)}")

    # Make sure all nodes have been assigned a partition.
    assert (all([n in node_map for n in mod.graph.nodes]))

    logger.debug("pre-fusion split mod:")
    logger.debug(
        lazy_graph_print_tabular(mod.graph, 'part', lambda x: node_map[x]))

    # Create subgraph for every fusable partition.
    subgraphs: Dict[torch.fx.Node, List[torch.fx.Node]] = dict()
    for n, p in node_map.items():
        if p > 0:
            if not p in subgraphs:
                subgraphs[p] = []
            subgraphs[p].append(n)

    logger.debug(f"Found {len(subgraphs)} fusable subgraphs.")

    # Attempt to fuse each subgraph.
    for p in subgraphs.keys():
        sub = SubGraph(fg, subgraphs[p])
        if len([n for n in sub.nodes if non_trivial_op(n)]) <= 1:
            logger.debug(f"Reject empty/singleton subgraph:\n{sub.tabular()}")
            continue
        logger.debug(
            f"Fusing sub-module (last_input={sub.last_input()}):\n{sub.tabular()}"
        )
        fuse_graph_nodes(cc, sub)
        logger.debug(f"Post fusion sub-module:\n{sub.tabular()}")

    logger.debug("Post fusion module:")
    logger.debug(lazy_graph_print_tabular(mod.graph))

    return mod