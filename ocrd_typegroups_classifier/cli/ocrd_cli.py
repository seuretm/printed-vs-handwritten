"""
OCR-D conformant command line interface
"""
import click
from ocrd.decorators import ocrd_cli_options, ocrd_cli_wrap_processor

from ..processor import TypegroupsClassifierProcessor

@click.command()
@ocrd_cli_options
def cli(*args, **kwargs):
    """
    Classify typegroups
    """
    return ocrd_cli_wrap_processor(TypegroupsClassifierProcessor, *args, **kwargs)
