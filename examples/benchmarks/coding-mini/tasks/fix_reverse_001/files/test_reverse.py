from reverse import reverse_string

def test_hello():
    assert reverse_string('hello') == 'olleh'

def test_empty():
    assert reverse_string('') == ''
