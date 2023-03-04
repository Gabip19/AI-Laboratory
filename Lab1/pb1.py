"""
Să se determine ultimul (din punct de vedere alfabetic) cuvânt care poate
apărea într-un text care conține mai multe cuvinte separate prin ” ” (spațiu).
De ex. ultimul (dpdv alfabetic) cuvânt din ”Ana are mere rosii si galbene” este cuvântul "si".
"""


def ultim_alfabetic1(text: str) -> str:
    '''
    O(n*logn)
    '''
    words_list = text.split(" ")
    sorted_words = sorted(words_list, reverse=True)
    return sorted_words[0]



def ultim_alfabetic2(text: str) -> str:
    '''
    O(n)
    '''
    words_list = text.split(" ")
    last_word = words_list[0]
    for i in range(1, len(words_list)):
        if words_list[i] > last_word:
            last_word = words_list[i]
    return last_word



def test(func):
    assert func("Ana are mere rosii si galbene") == "si"
    assert func("Ana are mere rosii galbene") == "rosii"
    assert func("cuvant") == "cuvant"
    assert func("ana ana") == "ana"
    assert func("multe cuvinte fara litere") == "multe"
    assert func("litere a b c z d a") == "z"


if __name__ == '__main__':
    test(ultim_alfabetic1)
    test(ultim_alfabetic2)
    rez = ultim_alfabetic2("Ana are mere rosii Si sir Sir aa z galbene")
    print(rez)