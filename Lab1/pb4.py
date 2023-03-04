"""
Să se determine cuvintele unui text care apar exact o singură dată în acel text.
De ex. cuvintele care apar o singură dată în "ana are ana are mere rosii ana"
sunt: 'mere' și 'rosii'.
"""


def single_occ1(text: str) -> list[str]:
    '''
    O(n^2)
    '''
    words = text.split(" ")
    single_occ = []
    
    for i in range(len(words)):
        ok = True
        for j in range(len(words)):
            if words[i] == words[j] and i != j:
                ok = False
        if ok == True:
            single_occ.append(words[i])
    
    return single_occ



def single_occ2(text: str) -> list[str]:
    '''
    O(n) timp
    O(n) spatiu
    '''
    words = text.split(" ")
    hash_table = {}
    for word in words:
        hash_table[word] = hash_table.get(word, 0) + 1
    
    single_occ = []
    for key in hash_table.keys():
        if hash_table[key] == 1:
            single_occ.append(key)
    
    return single_occ



def test(func):
    assert func("ana are ana are mere rosii ana") == ["mere", "rosii"]
    assert func("cuvant cuvinte cuvintele repet repet cuvant") == ["cuvinte", "cuvintele"]
    assert func("cuvant") == ["cuvant"]
    assert func("ana ana") == []
    assert func("propozitie propozitie care care nu se repeta") == ["nu", "se", "repeta"]
    assert func("litere a b c z d a b c d") == ["litere", "z"]


if __name__ == '__main__':
    test(single_occ1)
    test(single_occ2)
    rez = single_occ1("ana are ana are mere rosii ana")
    print(rez)