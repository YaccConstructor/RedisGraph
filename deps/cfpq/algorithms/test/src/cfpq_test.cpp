#include <cpu_graphblas.h>
#include <fstream>
#include <grammar.h>
#include <gtest/gtest.h>
#include <m4ri.h>
#include <sparse.h>
#include <triplet_loader.h>

using cfpq_t = int (*)(const Grammar *, CfpqResponse *, const GrB_Matrix *,
                       const char **, size_t, size_t);

class CFPQRun : public ::testing::TestWithParam<cfpq_t> {
public:
  ~CFPQRun() override {
    for (auto &[_, matrices] : graphs) {
      for (auto &[_, matrix] : matrices) {
        GrB_Matrix_free(&matrix);
      }
    }
  }

protected:
  std::map<std::string, std::map<std::string, GrB_Matrix>> graphs;
  std::map<std::string, Grammar> grammars;

  CfpqResponse run(const char *graph_name, const char *grammar_name,
                   cfpq_t func) {
    auto graph_it = graphs.find(graph_name);
    if (graph_it == graphs.end()) {
      std::ifstream file(graph_name);
      graph_it =
          graphs.insert(std::make_pair(graph_name, load_triplets(file))).first;
    }

    auto grammar_it = grammars.find(grammar_name);
    if (grammar_it == grammars.end()) {
      Grammar gr;
      FILE *f = fopen(grammar_name, "r");
      Grammar_Load(&gr, f);
      fclose(f);
      grammar_it = grammars.insert(std::make_pair(grammar_name, gr)).first;
    }

    std::vector<const char *> relations_names;
    std::vector<GrB_Matrix> relations;

    for (const auto &[name, relation] : graph_it->second) {
      relations_names.push_back(name.data());
      relations.push_back(relation);
    }

    CfpqResponse response;
    CfpqResponse_Init(&response);

    GrB_Index size;
    GrB_Matrix_nrows(&size, relations.front());

    func(&grammar_it->second, &response, relations.data(),
         relations_names.data(), relations.size(), size);

    return response;
  }
};

void check(const CfpqResponse &lhs,
           const std::map<std::string, GrB_Index> &rhs) {
  for (const auto &item : rhs) {
    auto idx = std::find_if(lhs.nonterms, lhs.nonterms + lhs.count,
                            [&](const char *it) {
                              return std::string_view(it) == item.first;
                            }) -
               lhs.nonterms;
    ASSERT_LT(idx, lhs.count);
    ASSERT_EQ(item.second, lhs.control_sums[idx]);
  }
  ASSERT_EQ(lhs.count, rhs.size());
}

class CFPQTestAll : public CFPQRun {};
INSTANTIATE_TEST_CASE_P(CFPQTestAll, CFPQTestAll,
                        ::testing::Values(sparse, cpu_graphblas, m4ri));

class CFPQTestAllWithoutM4ri : public CFPQRun {};
INSTANTIATE_TEST_CASE_P(CFPQTestAllWithoutM4ri, CFPQTestAllWithoutM4ri,
                        ::testing::Values(sparse, cpu_graphblas));

TEST_P(CFPQTestAll, SmallGraph) {
  auto solver = GetParam();
  auto response = run("resources/small/matrices/paper.txt",
                      "resources/small/grammars/paper.txt", solver);
  check(response, {{"s", 6}, {"s1", 6}, {"a", 3}, {"b", 2}});
}

TEST_P(CFPQTestAllWithoutM4ri, RdfGo) {
  auto solver = GetParam();
  auto response = run("resources/rdf/matrices/go.txt",
                      "resources/rdf/grammars/GPPerf1_cnf.txt", solver);
  check(response, {{"s", 304068},
                   {"s1", 90512},
                   {"s2", 90512},
                   {"s3", 58483},
                   {"s4", 58483},
                   {"s5", 278610},
                   {"s6", 39642}});
}
