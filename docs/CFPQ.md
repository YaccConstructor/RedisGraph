# Context-Free Path Querying (CFPQ)

CFPQ is a way to use context-free grammars as paths constraints by the same way as one can use regular expressions to specify sequences which can be formed by paths labels.

One of the classical example of context-free query is a *same-generation query* which can be expressed in terms of context-free grammar as follows:
```
s -> X X_R | X s X_R
```
where ```X``` is a some relation and ```X_R``` is a reversed relation. 
Thus, CFPQ allows one to explore hierarchical patterns in graph-structured data.

CFPQ can be use in different areas for graph-structured data analysis. Some examples of CFQP applications are listed below.
- Static code analysis
  - Taint analysis: [Scalable and Precise Taint Analysis for Android](http://huangw5.github.io/docs/issta15.pdf) 
  - Points-to analysis/alias analysis:
     - [An Incremental Points-to Analysis with CFL-Reachability](https://www.researchgate.net/publication/262173734_An_Incremental_Points-to_Analysis_with_CFL-Reachability)
     - [Graspan: A Single-machine Disk-based Graph System for Interprocedural Static Analyses of Large-scale Systems Code](https://dl.acm.org/doi/10.1145/3037697.3037744)
  - Binding time analysis: [BTA Termination Using CFL-Reachability](https://www.researchgate.net/publication/2467654_BTA_Termination_Using_CFL-Reachability)
- Graph segmentation: [Understanding Data Science Lifecycle Provenance via Graph Segmentation and Summarization](https://ieeexplore.ieee.org/abstract/document/8731467)
- Biological data analysis: [Subgraph queries by context-free grammars](https://www.researchgate.net/publication/321662505_Subgraph_Queries_by_Context-free_Grammars)


