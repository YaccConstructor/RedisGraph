import os
import sys
from RLTest import Env
from redisgraph import Graph, Node, Edge

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from base import FlowTestsBase


def create_pipe(redis_con):
    pipe = Graph("pipe", redis_con)
    node_props = ['v1', 'v2', 'v3', 'v4', 'v5']

    nodes = []
    for idx, v in enumerate(node_props):
        node = Node(label="L", properties={"val": v})
        nodes.append(node)
        pipe.add_node(node)

    edge = Edge(nodes[0], "A", nodes[1])
    pipe.add_edge(edge)

    edge = Edge(nodes[1], "A", nodes[2])
    pipe.add_edge(edge)

    edge = Edge(nodes[2], "B", nodes[3])
    pipe.add_edge(edge)

    edge = Edge(nodes[3], "B", nodes[4])
    pipe.add_edge(edge)

    pipe.commit()
    return pipe

def create_reversed_pipe(redis_con):
    reversed_pipe = Graph("reversed_pipe", redis_con)
    node_props = ['v1', 'v2', 'v3', 'v4', 'v5']

    nodes = []
    for idx, v in enumerate(node_props):
        node = Node(label="L", properties={"val": v})
        nodes.append(node)
        reversed_pipe.add_node(node)

    edge = Edge(nodes[0], "A", nodes[1])
    reversed_pipe.add_edge(edge)

    edge = Edge(nodes[2], "B", nodes[1])
    reversed_pipe.add_edge(edge)

    edge = Edge(nodes[3], "A", nodes[2])
    reversed_pipe.add_edge(edge)

    edge = Edge(nodes[3], "B", nodes[4])
    reversed_pipe.add_edge(edge)

    reversed_pipe.commit()
    return reversed_pipe


class testPathPattern(FlowTestsBase):
    def __init__(self):
        self.env = Env()
        redis_con = self.env.getConnection()
        self.pipe_graph = create_pipe(redis_con)
        self.reversed_pipe_graph = create_reversed_pipe(redis_con)

    def test00_path_pattern_explain(self):
        query = """
        PATH PATTERN S = ()-/ :A [~S | ()] :B /-()
        MATCH (a)-/ ~S /->(b)
        RETURN a.val, b.val ORDER BY a.val, b.val"""
        self.pipe_graph.execution_plan(query)

    def test01_path_pattern_explain(self):
        query = """
        PATH PATTERN S1 = ()-/ :A :B /->()
        PATH PATTERN S2 = ()-/ ~S1 :B /->()
        MATCH (a)-/ ~S2 /->(b)
        RETURN a, b"""
        self.pipe_graph.execution_plan(query)

    def test00_path_pattern_execution(self):
        query = """
        MATCH (a)-/ :A :B /->(b)
        RETURN a.val, b.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v2', 'v4']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test01_path_pattern_execution(self):
        query = """
        MATCH (a)-/ <[:A :B] /->(b)
        RETURN a.val, b.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v4', 'v2']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test02_path_pattern_execution(self):
        query = """
        MATCH (a)-/ <:B <:A /->(b)
        RETURN a.val, b.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v4', 'v2']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test03_path_pattern_execution(self):
        query = """
        MATCH (a)-/ <[<:A | :B] /->(b)
        RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v1', 'v2'], ['v2', 'v3'], ['v4', 'v3'], ['v5', 'v4']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test04_path_pattern_execution(self):
        query = """
        PATH PATTERN S = ()-/ :B /->()
        MATCH (a)-/ :A ~S /->(b)
        RETURN a, b ORDER BY a.val, b.val"""
        actual_result = self.pipe_graph.query(query)

    def test05_path_pattern_execution(self):
        query = """
        PATH PATTERN S = ()-/ [:A ~S :B] | [:A :B] /-()
        MATCH (a)-/ ~S /->(b)
        RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v1', 'v5'], ['v2', 'v4']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test06_path_pattern_execution(self):
        query = """MATCH (a)-/:A :A /->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v1', 'v3']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test07_path_pattern_execution(self):
        query = """MATCH (a)-/:A :B /->(b) RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v2', 'v4']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test08_path_pattern_execution(self):
        query = """
        PATH PATTERN S = ()-/ :A [~S | ()] :B /-()
        MATCH (a)-/ ~S /->(b)
        RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v1', 'v5'],
                           ['v2', 'v4']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test09_path_pattern_execution(self):
        query = """
        PATH PATTERN S = ()-/ :A [~S | ()] :B /->()
        MATCH (a)-[:A]-(b)-/ ~S /->(c)
        RETURN a.val, b.val, c.val ORDER BY a.val, b.val, c.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v1', 'v2', 'v4'],
                           ['v2', 'v1', 'v5'],
                           ['v3', 'v2', 'v4']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test10_path_pattern_execution(self):
        query = """
        PATH PATTERN S = ()-/ :A [~S | ()] :B /->()
        MATCH (a)-[:A]-(b)-/ ~S /->(c)-[:B]-(d)
        RETURN a.val, b.val, c.val, d.val ORDER BY a.val, b.val, c.val, d.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v1', 'v2', 'v4', 'v3'],
                           ['v1', 'v2', 'v4', 'v5'],
                           ['v2', 'v1', 'v5', 'v4'],
                           ['v3', 'v2', 'v4', 'v3'],
                           ['v3', 'v2', 'v4', 'v5']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test11_path_pattern_execution(self):
        query = """
        PATH PATTERN S = ()-/:A [~S | ()] :B/->()
        MATCH (a)-/ <~S /->(b)
        RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v4', 'v2'],
                           ['v5', 'v1']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test12_path_pattern_execution(self):
        query = """
        PATH PATTERN S = ()-/ :A [<~S | ()] :B /->()
        MATCH (a)-/ ~S /->(b)
        RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.reversed_pipe_graph.query(query)
        expected_result = [['v1', 'v5'],
                           ['v4', 'v2']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test13_path_pattern_execution(self):
        query = """
        PATH PATTERN S = ()-/ :A :B /->()
        MATCH (a)-/ <~S /->(b)
        RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v4', 'v2']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test14_path_pattern_execution(self):
        query = """
        MATCH (a)-/ :A /->(b)
        RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v2', 'v3']]
        self.env.assertEquals(actual_result.result_set, expected_result)
    
    def test15_path_pattern_execution(self):
        query = """
        MATCH (a)-/ :A | :B /->(b)
        RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v1', 'v2'],
                           ['v2', 'v3'],
                           ['v3', 'v4'],
                           ['v4', 'v5']]
        self.env.assertEquals(actual_result.result_set, expected_result)

    def test16_path_pattern_execution(self):
        query = """
        MATCH (a)-/ :A :B /->(b)
        RETURN a.val, b.val ORDER BY a.val, b.val"""
        actual_result = self.pipe_graph.query(query)
        expected_result = [['v2', 'v4']]
        self.env.assertEquals(actual_result.result_set, expected_result)