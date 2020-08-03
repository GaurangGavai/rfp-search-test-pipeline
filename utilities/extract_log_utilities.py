
import datetime

select_actions = ["CONTENT_COPY_TEXT", "CONTENT_FILE_DOWNLOAD", "CONTENT_PASTE"]
search_actions = ["CONTENT_SEARCH", "SEARCH_FROM_PROPOSAL"]
rejected_search_actions = ["CONTENT_BACK_TO_PROPOSAL", "PROPOSAL_CUSTOM_RESPONSE", "PROPOSAL_PASTE_TEXT"]
# TODO: We may eventually want to move proposal_paste_text to selectec search if it has enoughdata.
other = ["PROPOSAL_OPTION_SELECTION"]




def date_to_string(date):
    """Use this to define the canonical date representation that we'll use."""
    if isinstance(date, str):
        return date
    return date.strftime("%a %b %d %H:%M:%S %Y")

def parse_results_list(results_string):
    """
    :param results_string: a string in the form of '[id1, id2, id3, id4, ...]'
    :return: a list in the form of ['id1', 'id2', 'id3', 'id4', ...]
    """
    if type(results_string) is str:
        return results_string[1:-1].split(sep=", ")
    else:
        return []