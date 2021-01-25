import xml.etree.ElementTree as ET
from xml.dom import minidom
import sys

def parse_xml_with_comments(xml_path: str):
    """
    Parse XML while preserving comments.
        Input:
            xml_path: Path to XML file
        Outputs:
            tree    : Parsed tree
    """
    if sys.version_info[0]+0.1*sys.version_info[1]< 3.8:
        # python version < 3.8: Create a custom parser
        class CommentedTreeBuilder(ET.TreeBuilder):
            def comment(self, data):
                self.start(ET.Comment, {})
                self.data(data)
                self.end(ET.Comment)
        tree = ET.parse(xml_path, parser=ET.XMLParser(target=CommentedTreeBuilder()))
    else:
        # python version >= 3.8: Use default parser with corrent configuration
        parser_with_comments = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        tree = ET.parse(xml_path, parser=parser_with_comments)
    return tree


def get_xml_str(tree: ET.ElementTree=None,
                node:ET.Element=None,
                pretty=False):
    """
    Serealize tree/ node into string
        Input:
            tree        : ElementTree
            node        : Element
            pretty      : Pretty formatting
        Outputs:
            xml_str    : string
    """
    node = tree.getroot() if node is None else node
    # ET.dump(node)  # debug print
    xmlstr = ET.tostring(node, encoding='unicode', method='xml')
    if pretty:
        xmlstr = minidom.parseString(xmlstr).toprettyxml(indent="\t")
    return(xmlstr)


def merge_xmls( receiver_xml:str,
                donor_xml:str,
                receiver_node=None,
                donor_node=None,
                destination="str"):
    """
    Merge XMLs preserving MuJoCo structure
        Input:
            receiver_xml    : XML_filepath receiving data
            donor_xml       : XML_filepath to include
            receiver_node   : node_name where donor gets attached
            donor_node      : list of donor-nodes to attached (TODO)
            destination     : str / tree

        Output:
            merged_xml      : str or tree format
    """
    receiver_tree = parse_xml_with_comments(receiver_xml)
    receiver_elem = receiver_tree.find(receiver_node) if receiver_node else receiver_tree.getroot()
    assert receiver_elem, "Receiving node:{} not found".format(receiver_node)

    donor_tree = parse_xml_with_comments(donor_xml)
    donor_root = donor_tree.getroot()
    for child in donor_root:
        receiver_elem.append(child)

    if destination == "str":
        return get_xml_str(receiver_tree)
    elif destination == "tree":
        return receiver_tree


# test_xml = ET.parse("tendon_finger_v0.xml")
# print_xml(test_xml)
# xmlstr = merge_xmls(receiver_xml="tendon_finger_v0.xml",
#         donor_xml="tendon_finger_muscleAct.xml")

# with open("output_xmlstr.xml", "w") as f:
#     f.write(xmlstr)


# finger_tree = ET.parse("tendon_finger_v0.xml")
# foceAct_tree = ET.parse("tendon_finger_forceAct.xml")

# root = finger_tree.getroot()
# # for child in root:
#     # print(child.tag, child.attrib)

# wb = finger_tree.find('worldbody')
# include_root = foceAct_tree.getroot()
# for child in include_root:
#     root.append(child)

# ET.dump(root)
# # import ipdb; ipdb.set_trace()

