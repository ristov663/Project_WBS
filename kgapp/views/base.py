from urllib.parse import unquote


def decode_uri(uri):
    return unquote(uri.split('/')[-1])
