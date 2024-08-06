from sqlglot import exp
from sqlglot.expressions import Expression
import copy
from itertools import product
import sqlparse
import custom_ast
import random


class ASTNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children is not None else []
        self.visited = False

    def __eq__(self, other):
        if not isinstance(other, ASTNode):
            return False
        if self.value != other.value or len(self.children) != len(other.children):
            return False
        return all(c1 == c2 for c1, c2 in zip(self.children, other.children))

    def __hash__(self):
        return hash((self.value, tuple(hash(child) for child in self.children)))


def double_neg_transformation(node):
    if not isinstance(node, (exp.And, exp.Or)) or node is None or isinstance(node, exp.Not):
        return []

    transformed_trees = []
    not_node = exp.Not(this=node)
    double_not_node = exp.Not(this=not_node)
    transformed_trees.append(double_not_node)

    return transformed_trees


def de_morgans_transformation(node):
    if not isinstance(node, exp.And) and not isinstance(node, exp.Or):
        return []

    parent_node = node.parent
    if not isinstance(parent_node, exp.Not):
        return []

    transformed_trees = []
    if isinstance(node, exp.And):
        transformed_children = [exp.Not(this=grandchild) for grandchild in node.args.values()]
        new_or_node = exp.Or(this=transformed_children[0], expression=transformed_children[1])
        transformed_trees.append(new_or_node)
    elif isinstance(node, exp.Or):
        transformed_children = [exp.Not(this=grandchild) for grandchild in node.args.values()]
        new_and_node = exp.And(this=transformed_children[0], expression=transformed_children[1])
        transformed_trees.append(new_and_node)

    return transformed_trees


def comparison_transformation(node):
    if not isinstance(node, Expression) or node is None or node.args.get('this') is None:
        return []

    flip_map = {
        exp.EQ: exp.NEQ,
        exp.NEQ: exp.EQ,
        exp.LT: exp.GTE,
        exp.GT: exp.LTE,
        exp.LTE: exp.GT,
        exp.GTE: exp.LT
    }

    transformed_trees = []
    for original, flipped in flip_map.items():
        if isinstance(node, original):
            flipped_comparison = flipped(
                this=node.args['this'],
                expression=node.args['expression']
            )
            not_node = exp.Not(this=flipped_comparison)
            transformed_trees.append(not_node)
            #print("Eurphoria")


    return transformed_trees


def in_conversion_transform(node):
    if not isinstance(node, exp.In) or node is None or node.args.get('this') is None:
        return []

    transformed_trees = []
    column = node.args['this']
    try:
        values = node.args['expressions']
    except:
        return []

    if not values:
        return []

    or_node = exp.EQ(this=copy.deepcopy(column), expression=values[0])
    for value in values[1:]:
        comparison = exp.EQ(this=copy.deepcopy(column), expression=value)
        or_node = exp.Or(this=or_node, expression=comparison)

    transformed_trees.append(or_node)
    return transformed_trees




def generate_transformations_list(node):
    """Generate a list of each node in the tree along with each transformation that can be applied to it."""
    nodes_transformations = []

    def traverse_and_collect(node):
        if node is None:
            return

        transformations = {
            "double_negatives": double_neg_transformation(node),
            "de_morgans": de_morgans_transformation(node),
            "in_conversion": in_conversion_transform(node),
            "flip_comparison": comparison_transformation(node)
        }

        #num_de_morgans = len(transformations['de_morgans'])
        #print(f"Number of De Morgan's transformations available: {num_de_morgans}")

        nodes_transformations.append((node, transformations))

        if hasattr(node, 'args') and isinstance(node.args, dict):
            for child in node.args.values():
                if isinstance(child, exp.Expression):
                    traverse_and_collect(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, exp.Expression):
                            traverse_and_collect(item)


    traverse_and_collect(node)
    return nodes_transformations