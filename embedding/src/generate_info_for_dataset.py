# This file is used to print dataset info:
# How many triples; entities; attributes; are there in the dataset?
def toggle_file(folder, file):
    f_entities = open("../data/" + folder + "/entities.dict", "r")
    f_entity_lines = f_entities.readlines()
    f_co_attri = open("../data/" + folder + "/attributes.dict", "r")
    f_co_attri_lines = f_co_attri.readlines()
    entities_dict = []
    co_attri_dict = []
    for line1 in f_entity_lines:
        line_to_list1 = line1.split()
        entity1 = line_to_list1[1].rstrip('\n')
        entities_dict.append(entity1)
    for line2 in f_co_attri_lines:
        line_to_list2 = line2.split()
        co_attri = line_to_list2[1].rstrip('\n')
        co_attri_dict.append(co_attri)

    f_triple = open(file, "r")
    f_lines = f_triple.readlines()
    entity_list = []
    relation_list = []
    triple_list = []
    attri_list = []
    for line in f_lines:
        line_to_list = line.split()
        head = line_to_list[0]
        head_attri = co_attri_dict[entities_dict.index(head)]
        relation = line_to_list[1]
        tail = line_to_list[2].strip('\n')
        tail_attri = co_attri_dict[entities_dict.index(tail)]
        triple = (head, relation, tail)
        if head not in entity_list:
            entity_list.append(head)
        if relation not in relation_list:
            relation_list.append(relation)
        if tail not in entity_list:
            entity_list.append(tail)
        if triple not in triple_list:
            triple_list.append(triple)
        if head_attri not in attri_list:
            attri_list.append(head_attri)
        if tail_attri not in attri_list:
            attri_list.append(tail_attri)

    # print("entity: ", len(entity_list))
    # print("relation: ", len(relation_list))
    # print("triple: ", len(triple_list))
    print("attri: ", len(attri_list))


def generate_info(folder):
    print(folder)
    print("train:")
    toggle_file(folder, "../data/" + folder + "/train.txt")
    print("\n")
    print("test:")
    toggle_file(folder, "../data/" + folder + "/test.txt")
    print("\n")
    print("valid:")
    toggle_file(folder, "../data/" + folder + "/valid.txt")
    print("\n")


generate_info("fb237_v1_ind")
generate_info("fb237_v2_ind")
generate_info("fb237_v3_ind")
generate_info("fb237_v4_ind")

generate_info("nell_v1_ind")
generate_info("nell_v2_ind")
generate_info("nell_v3_ind")
generate_info("nell_v4_ind")

generate_info("wn18rr_v1_ind")
generate_info("wn18rr_v2_ind")
generate_info("wn18rr_v3_ind")
generate_info("wn18rr_v4_ind")
