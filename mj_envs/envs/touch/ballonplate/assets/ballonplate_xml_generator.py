from mj_envs.utils.xml_utils import parse_xml_with_comments, get_xml_str
import xml.etree.ElementTree as ET


def generate_ballonplate_xml(ballonplate_tempelate_xml,
                            x_plate_r = 0.1, y_plate_r = 0.1,
                            x_tac_n = 10, y_tac_n = 10,
                            x_tac_r = 0.008, y_tac_r = 0.008, z_tac_r = 0.001):

    """
    Use the template XML and add sites and touch sensors progmatically
    """

    # Parse tempelate xml
    xml_tree = parse_xml_with_comments(ballonplate_tempelate_xml)

    # gather relevant nodes
    root_elem = xml_tree.getroot()
    plate_elem = root_elem.find(".//body[@name='plate']")
    sensor_elem = root_elem.find("sensor")
    assert plate_elem is not None, "Plate body not found"

    # create sites
    for ix in range(x_tac_n):
        for iy in range(y_tac_n):
            # <site name="t0.0" type="box" size="0.01 0.01 0.001" pos="0 0 0.01"/>
            site = ET.SubElement(plate_elem, 'site')
            site.set('name', "s{}.{}".format(ix, iy))
            site.set('type', "box")
            site.set('size', "{} {} {}".format(x_tac_r, y_tac_r, z_tac_r))
            site.set('pos', "{:.4f} {:.4f} {:.4f}".format((2*ix+1)*x_plate_r/x_tac_n-x_plate_r, (2*iy+1)*y_plate_r/y_tac_n-y_plate_r, 0.01))

            # create touch sensors
            touch = ET.SubElement(sensor_elem, 'touch')
            touch.set('name', "t{}.{}".format(ix, iy))
            touch.set('site', "s{}.{}".format(ix, iy))
    return get_xml_str(tree = xml_tree, pretty=True)