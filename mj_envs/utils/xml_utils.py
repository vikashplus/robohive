import xml.etree.ElementTree as ET
from xml.dom import minidom
import sys

def parse_xml_with_comments(xml_path: str=None, xml_str: str=None):
    """
    Parse XML while preserving comments.
        Input:
            xml_path: Path to XML file
        Outputs:
            tree    : Parsed tree
    """

    if xml_str:
        tree = ET.ElementTree(ET.fromstring(xml_str))
    elif xml_path:
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
    else:
        raise TypeError("Both xml_path and xml_str can't be None")

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

    if pretty:
        # remove previous formatting
        node.tail = ""
        node.text = ""
        for elem in node.iter():
            elem.tail=""
            if elem.tag != ET.Comment:
                elem.text = ""
        xmlstr = ET.tostring(node, encoding='unicode', method='xml')
        xmlstr = minidom.parseString(xmlstr).toprettyxml(indent="\t")
    else:
        xmlstr = ET.tostring(node, encoding='unicode', method='xml')

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

def reassign_parent( xml_path: str=None,
                    xml_str: str=None,
                    receiver_node=None,
                    donor_node=None,
                    destination="str"):
    """
    Merge XMLs preserving MuJoCo structure
        Input:
            xml_path        : XML_filepath receiving data
            xml_str         : XML_str receiving data (higher priority over XML_filepath)
            receiver_node   : node_name where donor gets attached
            donor_node      : list of donor-nodes to attached (TODO)
            destination     : str / tree

        Output:
            merged_xml      : str or tree format
    """
    # import ipdb; ipdb.set_trace()

    # find elements
    xml_tree = parse_xml_with_comments(xml_path=xml_path, xml_str=xml_str)
    parent_elem = xml_tree.find(".//body[@name='{}']".format(receiver_node))
    assert parent_elem, "Parent node:{} not found".format(parent_elem)

    child_elem = xml_tree.find(".//body[@name='{}']".format(donor_node))
    assert child_elem, "Child node:{} not found".format(child_elem)

    child_parent_elem = xml_tree.find(".//body[@name='{}']...".format(donor_node))
    assert child_parent_elem, "Child's parent node:{} not found".format(child_elem)

    # move child
    parent_elem.append(child_elem)
    child_parent_elem.remove(child_elem)

    if destination == "str":
        return get_xml_str(xml_tree)
    elif destination == "tree":
        return xml_tree
