from decision_tree import *

if __name__ == "__main__":
    root_node = Node("root", all_examples)
    print("root node:", root_node.name)
    root_node.build_tree()
    root_node.print_tree()
