DEF_CONCEPTS = ('zebra', 'banana', 'pizza', 'umbrella', 'toothbrush', 'tooth brush')


def get_concepts():
    import os

    concepts = os.environ.get('CONCEPTS', None)
    if concepts is None:
        print('No concepts given from env variables, using default concepts')
        return DEF_CONCEPTS

    concepts = concepts.split(',')
    print(f'Using concepts: {concepts}')

    return concepts
