from src.data_parser.essays_parser import DataUnification
from src.data_parser.essays_parser import _ProcessExampleEssay, _ProcessSingleEssay


def ProcessExampleEssay():
    _ProcessExampleEssay()


def ProcessSingleEssay(full_text="NO TEXT AVAILABLE"):
    return _ProcessSingleEssay(full_text)


def ParseEssays():
    DataUnification()


def ParseWebDiscourse():
    # to be implemented
    pass


def ParseEarningCalls():
    pass
