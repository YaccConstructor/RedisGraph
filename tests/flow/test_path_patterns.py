import os
import sys
from RLTest import Env
from redisgraph import Graph, Node, Edge

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from base import FlowTestsBase

redis_graph = None

GRAPH_ID = "G"

class testPathPattern(FlowTestsBase):
    def __init__(self):
        self.env = Env()
        global redis_graph
        redis_con = self.env.getConnection()
        redis_graph = Graph(GRAPH_ID, redis_con)
        self.populate_graph()

    def populate_graph(self):
        node_props = ['v1', 'v2', 'v3', 'v4', 'v5']

        nodes = []
        for idx, v in enumerate(node_props):
            node = Node(label="L", properties={"val": v})
            nodes.append(node)
            redis_graph.add_node(node)

        edge = Edge(nodes[0], "A", nodes[1])
        redis_graph.add_edge(edge)

        edge = Edge(nodes[1], "A", nodes[2])
        redis_graph.add_edge(edge)

        edge = Edge(nodes[2], "B", nodes[3])
        redis_graph.add_edge(edge)

        edge = Edge(nodes[3], "B", nodes[4])
        redis_graph.add_edge(edge)

        redis_graph.commit()

    # def test00_path_pattern(self):
    #     query = """MATCH (a)-/:A :A/->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
    #     actual_result = redis_graph.query(query)
    #     expected_result = [['v1', 'v3']]
    #     self.env.assertEquals(actual_result.result_set, expected_result)
    #
    # def test01_path_pattern(self):
    #     query = """MATCH (a)-/:A :B/->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
    #     actual_result = redis_graph.query(query)
    #     expected_result = [['v2', 'v4']]
    #     self.env.assertEquals(actual_result.result_set, expected_result)
    #
    # def test02_path_pattern(self):
    #     query = """
    #     PATH PATTERN S = ()-/ :A [~S | ()] :B /-()
    #     MATCH (a)-/ ~S /->(b)
    #     RETURN a.val, b.val ORDER BY a.val, b.val"""
    #     actual_result = redis_graph.query(query)
    #     expected_result = [['v1', 'v5'],
    #                        ['v2', 'v4']]
    #     self.env.assertEquals(actual_result.result_set, expected_result)

    def test00_path_pattern_explain(self):
        query = """
        PATH PATTERN S = ()-/ :A [~S | ()] :B /-()
        MATCH (a)-/ ~S /->(b)
        RETURN a.val, b.val ORDER BY a.val, b.val"""
        redis_graph.execution_plan(query)

    def test01_path_pattern_explain(self):
        query = """
        PATH PATTERN S1 = ()-/ :A :B /->()
        PATH PATTERN S2 = ()-/ ~S1 :B /->()
        MATCH (a)-/ ~S2 /->(b)
        RETURN a, b"""
        redis_graph.execution_plan(query)

    def test00_path_pattern_execution(self):
        query = """
        MATCH (a)-/ :A :B /->(b)
        RETURN a.val, b.val"""
        actual_result = redis_graph.query(query)
        expected_result = [['v2', 'v4']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test01_path_pattern_execution(self):
        query = """
        MATCH (a)-/ <[:A :B] /->(b)
        RETURN a.val, b.val"""
        actual_result = redis_graph.query(query)
        expected_result = [['v4', 'v2']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test02_path_pattern_execution(self):
        query = """
        MATCH (a)-/ <:B <:A /->(b)
        RETURN a.val, b.val"""
        actual_result = redis_graph.query(query)
        expected_result = [['v4', 'v2']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test03_path_pattern_execution(self):
        query = """
        MATCH (a)-/ <[<:A | :B] /->(b)
        RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = redis_graph.query(query)
        expected_result = [['v1', 'v2'], ['v2', 'v3'], ['v4', 'v3'], ['v5', 'v4']]
        self.env.assertEquals(actual_result.result_set, expected_result)
