from json import loads

from pkg_resources import resource_string

classes = {
    0: "Antiqua",
    1: "Bastarda",
    6: "Rotunda",
    7: "Textura"
}

OCRD_TOOL = loads(resource_string(__name__, 'ocrd-tool.json'))
