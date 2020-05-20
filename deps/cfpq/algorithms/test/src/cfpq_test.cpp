#include <chrono>
#include <cpu_graphblas.h>
#include <fstream>
#include <grammar.h>
#include <gtest/gtest.h>
#include <nsparse.h>
#include <sparse.h>
#include <triplet_loader.h>
#include <cusparse_cfpq.h>

using cfpq_t = int (*)(const Grammar*, CfpqResponse*, const GrB_Matrix*, const char**, size_t,
                       size_t);

void init_gb() {
  static bool init = false;
  if (!init) {
    GrB_init(GrB_NONBLOCKING);
  }
  init = true;
}

std::map<std::string, std::map<std::string, GrB_Matrix>> graphs;
std::map<std::string, Grammar> grammars;

class CFPQRun : public ::testing::TestWithParam<cfpq_t> {
 protected:
  CfpqResponse run(const char* graph_name, const char* grammar_name, cfpq_t func) {
    init_gb();

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

    for (auto i = 0; i < 3; i++) {
      response.time_to_prepare = 0;
      response.time_to = 0;
      response.count = 0;
      response.iteration_count = 0;
      func(&grammar_it->second, &response, relations.data(), relations_names.data(),
           relations.size(), size);
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Nodes: " << size << std::endl;
    std::cout << "Eval time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 3 << " ms"
              << std::endl;

    return response;
  }
};

void check(const CfpqResponse& lhs, const std::map<std::string, GrB_Index>& rhs) {
  std::cout << "Time to prepare: " << int(lhs.time_to_prepare * 1000) << " ms" << std::endl;
  std::cout << "Time to part: " << int(lhs.time_to * 1000) << " ms" << std::endl;
  std::cout << "Iterations: " << lhs.iteration_count << std::endl;

  for (auto i = 0; i < lhs.count; i++) {
    std::cout << lhs.nonterms[i] << ": " << lhs.control_sums[i] << std::endl;
  }

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
                        ::testing::Values(cpu_graphblas, sparse, nsparse_cfpq));

class CFPQTestNsparse : public CFPQRun {};
INSTANTIATE_TEST_CASE_P(CFPQTestNsparse, CFPQTestNsparse,
                        ::testing::Values(nsparse_cfpq_index));

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

TEST_P(CFPQTestNsparse, RdfPathwaysG1) {
  auto solver = GetParam();
  auto response = run(
      "/home/jblab/CFPQ-with-RedisGraph/akhoroshev/CFPQ_Data/data/graphs/RDF/Matrices/pathways.txt",
      "resources/rdf/grammars/GPPerf1_cnf.txt", solver);
  check(response, {});
}

TEST_P(CFPQTestNsparse, RdfPathwaysG2) {
  auto solver = GetParam();
  auto response = run(
      "/home/jblab/CFPQ-with-RedisGraph/akhoroshev/CFPQ_Data/data/graphs/RDF/Matrices/pathways.txt",
      "resources/rdf/grammars/GPPerf2_cnf.txt", solver);
  check(response, {});
}

TEST_P(CFPQTestNsparse, RdfGoHierarchyG1) {
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

TEST_P(CFPQTestNsparse, RdfGoHierarchyG2) {
  auto solver = GetParam();
  auto response = run("resources/rdf/matrices/go-hierarchy.txt",
                      "resources/rdf/grammars/GPPerf2_cnf.txt", solver);
  check(response, {});
}

TEST_P(CFPQTestNsparse, RdfEnzymeG1) {
  auto solver = GetParam();
  auto response = run(
      "/home/jblab/CFPQ-with-RedisGraph/akhoroshev/CFPQ_Data/data/graphs/RDF/Matrices/enzyme.txt",
      "resources/rdf/grammars/GPPerf1_cnf.txt", solver);
  check(response, {});
}

TEST_P(CFPQTestNsparse, RdfEnzymeG2) {
  auto solver = GetParam();
  auto response = run(
      "/home/jblab/CFPQ-with-RedisGraph/akhoroshev/CFPQ_Data/data/graphs/RDF/Matrices/enzyme.txt",
      "resources/rdf/grammars/GPPerf2_cnf.txt", solver);
  check(response, {});
}

TEST_P(CFPQTestNsparse, RdfEclassG1) {
  auto solver = GetParam();
  auto response =
      run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/CFPQ_Data/data/graphs/RDF/Matrices/"
          "eclass_514en.txt",
          "resources/rdf/grammars/GPPerf1_cnf.txt", solver);
  check(response, {});
}

TEST_P(CFPQTestNsparse, RdfEclassG2) {
  auto solver = GetParam();
  auto response =
      run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/CFPQ_Data/data/graphs/RDF/Matrices/"
          "eclass_514en.txt",
          "resources/rdf/grammars/GPPerf2_cnf.txt", solver);
  check(response, {});
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

TEST_P(CFPQTestNsparse, RdfTaxGPPerf1) {
  auto solver = GetParam();
  auto response =
      run("/mnt/data/tmp/taxonomy.txt", "resources/rdf/grammars/GPPerf1_cnf.txt", solver);
  check(response, {
                      {"s", 6977582},
                      {"s1", 35583954},
                      {"s2", 35583954},
                      {"s3", 0},
                      {"s4", 0},
                      {"s5", 4531241},
                      {"s6", 0},
                  });
}

TEST_P(CFPQTestNsparse, RdfTaxGPPerf2) {
  auto solver = GetParam();
  auto response =
      run("/mnt/data/tmp/taxonomy.txt", "resources/rdf/grammars/GPPerf2_cnf.txt", solver);
  check(response, {
                      {"b", 36722046},
                      {"s1", 35583954},
                      {"s2", 35583954},
                      {"s3", 33697004},
                  });
}

 TEST_P(CFPQTestNsparse, RdfTaxHieGPPerf1) {
  auto solver = GetParam();
  auto response = run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/taxonomy-hie.txt",
                      "resources/rdf/grammars/GPPerf1_cnf.txt", solver);
  check(response, {
                      {"s", 6977582},
                      {"s1", 35583954},
                      {"s2", 35583954},
                      {"s3", 0},
                      {"s4", 0},
                      {"s5", 4531241},
                      {"s6", 0},
                  });
}

 TEST_P(CFPQTestNsparse, RdfTaxHieGPPerf2) {
  auto solver = GetParam();
  auto response = run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/taxonomy-hie.txt",
                      "resources/rdf/grammars/GPPerf2_cnf.txt", solver);
  check(response, {
                      {"b", 36722046},
                      {"s1", 35583954},
                      {"s2", 35583954},
                      {"s3", 33697004},
                  });
}

// FreeScale
// 10k
 TEST_P(CFPQTestNsparse, FreeScale10000_3) {
  auto solver = GetParam();
  auto response =
      run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/Matrices/free_scale_graph_10000_3.txt",
          "resources/freescale/grammars/an_bm_cm_dn.txt", solver);
  check(response, {});
}

 TEST_P(CFPQTestNsparse, FreeScale10000_5) {
  auto solver = GetParam();
  auto response =
      run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/Matrices/free_scale_graph_10000_5.txt",
          "resources/freescale/grammars/an_bm_cm_dn.txt", solver);
  check(response, {});
}

 TEST_P(CFPQTestNsparse, FreeScale10000_10) {
  auto solver = GetParam();
  auto response =
      run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/Matrices/free_scale_graph_10000_10.txt",
          "resources/freescale/grammars/an_bm_cm_dn.txt", solver);
  check(response, {});
}

// 25k
 TEST_P(CFPQTestNsparse, FreeScale25000_3) {
  auto solver = GetParam();
  auto response =
      run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/Matrices/free_scale_graph_25000_3.txt",
          "resources/freescale/grammars/an_bm_cm_dn.txt", solver);
  check(response, {});
}

 TEST_P(CFPQTestNsparse, FreeScale25000_5) {
  auto solver = GetParam();
  auto response =
      run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/Matrices/free_scale_graph_25000_5.txt",
          "resources/freescale/grammars/an_bm_cm_dn.txt", solver);
  check(response, {});
}

 TEST_P(CFPQTestNsparse, FreeScale25000_10) {
  auto solver = GetParam();
  auto response =
      run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/Matrices/free_scale_graph_25000_10.txt",
          "resources/freescale/grammars/an_bm_cm_dn.txt", solver);
  check(response, {});
}


// 50k
 TEST_P(CFPQTestNsparse, FreeScale50000_3) {
  auto solver = GetParam();
  auto response =
      run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/Matrices/free_scale_graph_50000_3.txt",
          "resources/freescale/grammars/an_bm_cm_dn.txt", solver);
  check(response, {});
}

 TEST_P(CFPQTestNsparse, FreeScale50000_5) {
  auto solver = GetParam();
  auto response =
      run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/Matrices/free_scale_graph_50000_5.txt",
          "resources/freescale/grammars/an_bm_cm_dn.txt", solver);
  check(response, {});
}

 TEST_P(CFPQTestNsparse, FreeScale50000_10) {
  auto solver = GetParam();
  auto response =
      run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/Matrices/free_scale_graph_50000_10.txt",
          "resources/freescale/grammars/an_bm_cm_dn.txt", solver);
  check(response, {});
}

// 100k
 TEST_P(CFPQTestNsparse, FreeScale100000_3) {
  auto solver = GetParam();
  auto response =
      run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/Matrices/free_scale_graph_100000_3.txt",
          "resources/freescale/grammars/an_bm_cm_dn.txt", solver);
  check(response, {});
}

 TEST_P(CFPQTestNsparse, FreeScale100000_5) {
  auto solver = GetParam();
  auto response =
      run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/Matrices/free_scale_graph_100000_5.txt",
          "resources/freescale/grammars/an_bm_cm_dn.txt", solver);
  check(response, {});
}


 TEST_P(CFPQTestNsparse, FreeScale100000_10) {
  auto solver = GetParam();
  auto response =
      run("/home/jblab/CFPQ-with-RedisGraph/akhoroshev/Matrices/free_scale_graph_100000_10.txt",
          "resources/freescale/grammars/an_bm_cm_dn.txt", solver);
  check(response, {});
}
