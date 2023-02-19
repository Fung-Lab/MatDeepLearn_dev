from itertools import combinations, product


def subsets(arr: set) -> list:
    subsets = []
    [subsets.extend(list(combinations(arr, n))) for n in range(len(arr) + 1)]
    return subsets[1:]


def generate_mp_combos(mp_attrs: dict, num_layers) -> list:
    return [
        list([list(y) for y in x])
        for x in product(subsets(mp_attrs), repeat=num_layers)
    ]


if __name__ == "__main__":
    mp_attrs = ["rr", "rv"]
    num_layers = 4
    all_mp_combos = list(generate_mp_combos(mp_attrs, num_layers))
    print(all_mp_combos)
