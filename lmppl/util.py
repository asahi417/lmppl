import urllib


def internet_connection(host: str = 'http://google.com'):
    """ check if internet connection is available """
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False
