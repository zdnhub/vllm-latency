from FlagEmbedding import FlagReranker

reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
score = reranker.compute_score([("hello world", "nice to meet you"),
                                ("head north", "head south")])
print(score)
