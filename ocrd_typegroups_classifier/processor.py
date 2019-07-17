"""
Wrap TypegroupsClassifier as an ocrd.Processor
"""

from ocrd import Processor
from ocrd_utils import getLogger, concat_padded, MIMETYPE_PAGE
from ocrd_models.ocrd_page import (
    to_xml,

    TextStyleType
)
from ocrd_modelfactory import page_from_file

from .typegroups_classifier import TypegroupsClassifier
from .constants import OCRD_TOOL


class TypegroupsClassifierProcessor(Processor):

    def __init__(self, *args, **kwargs):
        kwargs['ocrd_tool'] = OCRD_TOOL['tools']['ocrd-typegroups-classifier']
        kwargs['version'] = OCRD_TOOL['version']
        super(TypegroupsClassifierProcessor, self).__init__(*args, **kwargs)
        self.log = getLogger('ocrd_typegroups_classifier')

    def process(self):
        network_file = self.parameter['network']
        stride = self.parameter['stride']
        classifier = TypegroupsClassifier.load(network_file)

        ignore_type = ('Adornment', 'Book covers and other irrelevant data',
                       'Empty Pages', 'Woodcuts - Engravings')

        self.log.debug('Processing: %s', self.input_files)
        for (_, input_file) in enumerate(self.input_files):
            pcgts = page_from_file(self.workspace.download_file(input_file))
            image_url = pcgts.get_Page().imageFilename
            pil_image = self.workspace.resolve_image_as_pil(image_url)
            result = classifier.run(pil_image, stride)
            score_sum = 0
            for typegroup in classifier.classMap.cl2id:
                if not typegroup in ignore_type:
                    score_sum += max(0, result[typegroup])

            script_highscore = 0
            noise_highscore = 0
            result_map = {}
            output = ''
            for typegroup in classifier.classMap.cl2id:
                score = result[typegroup]
                if typegroup in ignore_type:
                    noise_highscore = max(noise_highscore, score)
                else:
                    script_highscore = max(script_highscore, score)
                    normalised_score = max(0, score / score_sum)
                    result_map[normalised_score] = typegroup
            if noise_highscore > script_highscore:
                pcgts.get_Page().set_primaryScript(None)
                self.log.debug(
                    'Detected only noise (such as empty page or book cover). noise_highscore=%s > script_highscore=%s',
                    noise_highscore, script_highscore)
            else:
                for k in sorted(result_map, reverse=True):
                    if output != '':
                        output = '%s, ' % output
                    output = '%s%s:%d' % (output, result_map[k], round(100*k))
                self.log.debug('Detected %s' % output)
                page = pcgts.get_Page()
                textStyle = page.get_TextStyle()
                if not textStyle:
                    textStyle = TextStyleType()
                    page.set_TextStyle(textStyle)
                textStyle.set_fontFamily(output)
                ID = concat_padded(self.output_file_grp, input_file.ID)
                self.workspace.add_file(
                    ID=ID,
                    file_grp=self.output_file_grp,
                    mimetype=MIMETYPE_PAGE,
                    local_filename="%s/%s" % (self.output_file_grp, ID),
                    content=to_xml(pcgts)
                )
