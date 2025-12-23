import re

def process_xlsum(dataset):
    """
    Process XLSum dataset by cleaning up text and summary fields.
    Removes any double spaces from both fields using regex substitution.
    """
    def _process_doc(doc):
        # Remove double spaces
        doc["text"] = re.sub(r" +", " ", doc["text"])
        doc["summary"] = re.sub(r" +", " ", doc["summary"])
        return doc

    return dataset.map(_process_doc)
