#include <chrono>
#include <cpu_graphblas.h>
#include <fstream>
#include <grammar.h>
#include <gtest/gtest.h>
#include <m4ri.h>
#include <nsparse.h>
#include <sparse.h>
#include <triplet_loader.h>

using cfpq_t = int (*)(const Grammar*, CfpqResponse*, const GrB_Matrix*, const char**, size_t,
                       size_t);

class CFPQRun : public ::testing::TestWithParam<cfpq_t> {
 public:
  ~CFPQRun() override {
    for (auto& [_, matrices] : graphs) {
      for (auto& [_, matrix] : matrices) {
        GrB_Matrix_free(&matrix);
      }
    }
  }

 protected:
  std::map<std::string, std::map<std::string, GrB_Matrix>> graphs;
  std::map<std::string, Grammar> grammars;

  CfpqResponse run(const char* graph_name, const char* grammar_name, cfpq_t func) {
    auto graph_it = graphs.find(graph_name);
    if (graph_it == graphs.end()) {
      std::ifstream file(graph_name);
      graph_it = graphs.insert(std::make_pair(graph_name, load_triplets(file))).first;
    }

    auto grammar_it = grammars.find(grammar_name);
    if (grammar_it == grammars.end()) {
      Grammar gr;
      FILE* f = fopen(grammar_name, "r");
      Grammar_Load(&gr, f);
      fclose(f);
      grammar_it = grammars.insert(std::make_pair(grammar_name, gr)).first;
    }

    std::vector<const char*> relations_names;
    std::vector<GrB_Matrix> relations;

    for (const auto& [name, relation] : graph_it->second) {
      relations_names.push_back(name.data());
      relations.push_back(relation);
    }

    CfpqResponse response;
    CfpqResponse_Init(&response);

    GrB_Index size;
    GrB_Matrix_nrows(&size, relations.front());

    auto t1 = std::chrono::high_resolution_clock::now();

    func(&grammar_it->second, &response, relations.data(), relations_names.data(), relations.size(),
         size);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Nodes: " << size << std::endl;
    std::cout << "Eval time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() << " ms"
              << std::endl;

    return response;
  }
};

void check(const CfpqResponse& lhs, const std::map<std::string, GrB_Index>& rhs) {
  std::cout << "Time to prepare: " << int(lhs.time_to_prepare * 1000) << " ms" << std::endl;
  std::cout << "Iterations: " << lhs.iteration_count << std::endl;

  for (const auto& item : rhs) {
    auto idx = std::find_if(lhs.nonterms, lhs.nonterms + lhs.count,
                            [&](const char* it) { return std::string_view(it) == item.first; }) -
               lhs.nonterms;
    ASSERT_LT(idx, lhs.count);
    ASSERT_EQ(item.second, lhs.control_sums[idx]);
  }
  ASSERT_EQ(lhs.count, rhs.size());
}

class CFPQTestAll : public CFPQRun {};
INSTANTIATE_TEST_CASE_P(CFPQTestAll, CFPQTestAll,
                        ::testing::Values(cpu_graphblas, sparse, m4ri, nsparse_cfpq));

class CFPQTestNsparse : public CFPQRun {};
INSTANTIATE_TEST_CASE_P(CFPQTestNsparse, CFPQTestNsparse,
                        ::testing::Values(nsparse_cfpq, cpu_graphblas));

TEST_P(CFPQTestNsparse, SmallGraph) {
  auto solver = GetParam();
  auto response =
      run("resources/small/matrices/paper.txt", "resources/small/grammars/paper.txt", solver);
  check(response, {
                      {"s", 6},
                      {"s1", 6},
                      {"a", 3},
                      {"b", 2},
                  });
}

TEST_P(CFPQTestNsparse, RdfSkosGPPerf1) {
  auto solver = GetParam();
  auto response =
      run("resources/rdf/matrices/skos.txt", "resources/rdf/grammars/GPPerf1_cnf.txt", solver);
  check(response, {
                      {"s", 810},
                      {"s1", 1},
                      {"s2", 1},
                      {"s3", 70},
                      {"s4", 70},
                      {"s5", 5},
                      {"s6", 0},
                  });
}

TEST_P(CFPQTestNsparse, RdfGoGPPerf1) {
  auto solver = GetParam();
  auto response =
      run("resources/rdf/matrices/go.txt", "resources/rdf/grammars/GPPerf1_cnf.txt", solver);
  check(response, {
                      {"s", 304070},
                      {"s1", 90512},
                      {"s2", 90512},
                      {"s3", 58483},
                      {"s4", 58483},
                      {"s5", 278610},
                      {"s6", 39642},
                  });
}

TEST_P(CFPQTestNsparse, RdfGoGPPerf2) {
  auto solver = GetParam();
  auto response =
      run("resources/rdf/matrices/go.txt", "resources/rdf/grammars/GPPerf2_cnf.txt", solver);
  check(response, {
                      {"b", 334850},
                      {"s1", 90512},
                      {"s2", 90512},
                      {"s3", 327628},
                  });
}

TEST_P(CFPQTestNsparse, RdfGoHierarchy) {
  auto solver = GetParam();
  auto response = run("resources/rdf/matrices/go-hierarchy.txt",
                      "resources/rdf/grammars/GPPerf1_cnf.txt", solver);
  check(response, {
                      {"s", 588976},
                      {"s1", 490109},
                      {"s2", 490109},
                      {"s3", 0},
                      {"s4", 0},
                      {"s5", 324016},
                      {"s6", 0},
                  });
}

TEST_P(CFPQTestNsparse, RdfGeospecies) {
  auto solver = GetParam();
  auto response =
      run("resources/rdf/matrices/geospeices.txt", "resources/rdf/grammars/geo.cnf", solver);
  check(response, {
                      {"s", 226669749},
                      {"bt", 20867},
                      {"s1", 21361542},
                      {"btr", 20867},
                  });
}

TEST_P(CFPQTestNsparse, Sg5k) {
  auto solver = GetParam();
  auto response =
      run("resources/sg/matrices/G5k-0.001.txt", "resources/sg/grammars/SG.txt", solver);
  check(response, {
                      {"s", 24730729},
                      {"s1", 25046},
                      {"s2", 25046},
                      {"s3", 24730729},
                  });
}

TEST_P(CFPQTestNsparse, Sg10k) {
  auto solver = GetParam();
  auto response =
      run("resources/sg/matrices/G10k-0.001.txt", "resources/sg/grammars/SG.txt", solver);
  check(response, {
                      {"s", 100000000},
                      {"s1", 99805},
                      {"s2", 99805},
                      {"s3", 100000000},
                  });
}

TEST_P(CFPQTestNsparse, Sg10k0dot01) {
  auto solver = GetParam();
  auto response =
      run("resources/sg/matrices/G10k-0.01.txt", "resources/sg/grammars/SG.txt", solver);
  check(response, {
                      {"s", 100000000},
                      {"s1", 498331},
                      {"s2", 501170},
                      {"s3", 100000000},
                  });
}

TEST_P(CFPQTestNsparse, WorstCase512) {
  auto solver = GetParam();
  auto response = run("resources/worstcase/matrices/worstcase_512.txt",
                      "resources/worstcase/grammars/Brackets.txt", solver);
  check(response, {
                      {"s", 65792},
                      {"a", 257},
                      {"b", 256},
                      {"s1", 65792},
                  });
}

TEST_P(CFPQTestNsparse, FreeScale10000_5) {
  auto solver = GetParam();
  auto response = run("resources/freescale/matrices/free_scale_graph_10000_5.txt",
                      "resources/freescale/grammars/an_bm_cm_dn.txt", solver);
  check(response, {
                      {"s", 51353},
                      {"A", 12602},
                      {"S1", 40617},
                      {"D", 12584},
                      {"S2", 41557},
                      {"X", 51382},
                      {"B", 12447},
                      {"C", 12367},
                      {"X1", 45231},
                  });
}

TEST_P(CFPQTestNsparse, FreeScale10000_10) {
  auto solver = GetParam();
  auto response = run("resources/freescale/matrices/free_scale_graph_10000_10.txt",
                      "resources/freescale/grammars/an_bm_cm_dn.txt", solver);
  check(response, {
                      {"s", 669067},
                      {"A", 24984},
                      {"S1", 571345},
                      {"D", 24948},
                      {"S2", 516483},
                      {"X", 493102},
                      {"B", 25105},
                      {"C", 24963},
                      {"X1", 476972},
                  });
}