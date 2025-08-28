from utils import read_entity_description
import os

class Span_patch_generator():
    def __init__(self, span_threadshold = 0.5, data_name = "GMNER"):
        self.span_threadshold = span_threadshold
        self.data_name = data_name

    def get_system_prompt(self):
        system_prompt = self.Role_Definition()  + self.Format_output_Description() + self.Action_Definition()
        return system_prompt

    def Role_Definition(self):

        role_describe =  "[Role]\n " \
                        "You are an AI assistant focused on correcting textual named entity.\n" \
                        
        return role_describe

    def Action_Definition(self):
        action_describe = " [Action Required]\n" \
                        " - Carefully review the pre-detect textual named entity.\n"\
                        " - Please think step by step:\n " \
                        "1. What is the background knowledge of \"entity\" according to the original text? \n"\
                        "2. Is the span of the pre-detected entity correct? If not, what should the correct span be?\n"\
                        " - If you think the span of the entity is inaccurate, please correct the boundary of its span. Otherwise, just output the original prediction.\n"\
                        " - Important note: When you are correcting the span of an entity, please focus on tiny boundaries modification (one or two words) around the span and do not have additional outputs. \n"\
                        " - Please output your reasoning process and your final Corrected entity according to [Format_output_Description].\n" \

        return action_describe

    def Format_output_Description(self):

        format_describe = "[Format_output_Description]:\n" \
                        "eg: ```json{\"reasoning_process\": \"....\", \"corrected_entity\": \"....\"}```\n"
        
        return format_describe


class Region_patch_generator():
    def __init__(self, region_threadshold = 0.5, data_name = "CCKS", entity_type_desc_path= "./entity_type_desc.txt"):
        self.span_threadshold = region_threadshold
        self.data_name = data_name
        self.entity_type_desc_path = entity_type_desc_path

    def get_system_prompt(self):
        system_prompt = self.Role_Definition() + self.Entity_Type_Description()  + self.Action_Definition() + self.Format_output_Description() 
        return system_prompt

    def Role_Definition(self):

        role_describe =  "[Role]\n " \
                        "You are an AI assistant focused on correcting some errors of the given entities and detecting their corresponding visual bounding box within the provided image in military domain.\n" \

        role_describe_wo_refine =  "[Role]\n " \
                        "You are an AI assistant focused on locating visual entity bounding boxes within the provided image in military domain.\n" \


        return role_describe_wo_refine

    def Action_Definition(self):

        action_describe = " [Action Required]\n" \
                        " - Carefully review the provided image, text and the entities from **[Original Input]**\n"\
                        " - Please follow the instructions below step by step:\n " \
                        "1. Double check the correctness of the given **entity** in text according to the [Entity Type Description]. Fix these errors, including their **name** and **type**. Only make corrections when you are **very confident** on it. **DO NOT** generate any extra entities.\n"\
                        "2. For each entity, identify its corresponding absolute pixel bounding boxes [xmin, ymin, xmax, ymax] in the image. \n"\
                        " - **Important note**: (1) \"location\" entities cannot be grounded, please output [] for their bounding box. (2) If the entity cannot be precisely located at a specific bounding box within the image, set its bounding box to []. (3) One entity may match multiple bounding boxes\n"\
                        " - Please output your final results according to [Format_output_Description].\n" \
        
        action_describe_wo_refine = " [Action Required]\n" \
                        " Carefully review the provided image, text and the entities from **[Original Input]**\n"\
                        " - Please follow the instructions below:\n " \
                        "For each entity, identify its corresponding absolute pixel bounding boxes [xmin, ymin, xmax, ymax] in the image. Please refer to a detailed [Entity Type Description] and the background knowledge you know. \n"\
                        " - **Important note**: (1) \"location\" entities cannot be grounded, please output [] for their bounding box. (2) If the entity cannot be precisely located at a specific bounding box within the image, set its bounding box to []. (3) One entity may match multiple bounding boxes.\n"\
                        " - Please output your final results according to [Format_output_Description].\n" \

        return action_describe_wo_refine

    def Entity_Type_Description(self):

        if 'CCKS' in self.data_name:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            abs_entity_type_desc_path = os.path.join(current_dir, self.entity_type_desc_path)
            label_describe = read_entity_description(abs_entity_type_desc_path)
        else:
            raise Exception("Invalid dataname! Please check")

        return label_describe


    def Format_output_Description(self):

        format_describe = "[Format_output_Description]:\n" \
                        "eg: ```json\n [{\"entity\": \"Boeing E-3\", \"entity_type\": \"aircraft\", \"bounding_box\": [293, 21, 593, 449]}, {\"entity\": \"G36k\", \"entity_type\": \"weapon\", \"bounding_box\": [31, 94, 564, 612]}, ...]\n```"
        
        return format_describe


class MLLM_base_generator():
    def __init__(self, data_name = "GMNER", entity_type_desc_path = "./entity_type_desc.txt"):
        self.data_name = data_name
        self.entity_type_desc_path = entity_type_desc_path
    def get_system_prompt(self):
        system_prompt = self.Role_Definition() + self.Entity_Type_Description()  +  self.Action_Definition() + self.Format_output_Description() 
        return system_prompt

    def Role_Definition(self):

        role_describe =  "[Role]\n " \
                        "You are an AI assistant focused on extracting the textual named entity and their corresponding bounding box from the provided image-text pair in military domain.\n" \
                        
        return role_describe

    def Action_Definition(self):

        action_describe = " [Action Required]\n" \
                        " - Carefully review the provided image and the original text. \n"\
                        " - Please follow the instructions below step by step:\n " \
                        "1. Identity all the named entities in the original text according to the [Entity Type Description].\n"\
                        "2. For each named entity with its Entity Type, identify its corresponding absolute pixel bounding box [xmin, ymin, xmax, ymax] in the image.\n"\
                        " - **Important note**: (1) **DO NOT** extract entities without specific name (e.g.: gun, air base, fighter jet, weapon, soldier) (2) \"location\" entities cannot be grounded, please output [] for their bounding box. (3) If the entity cannot be precisely located at a specific bounding box within the image, set its bounding box to []. (4) Only output one nearest Entity Type.\n"\
                        " - Please output your final results according to [Format_output_Description].\n" \
        
        return action_describe

    def Entity_Type_Description(self):

        if 'CCKS' in self.data_name:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            abs_entity_type_desc_path = os.path.join(current_dir, self.entity_type_desc_path)
            label_describe = read_entity_description(abs_entity_type_desc_path)
        else:
            raise Exception("Invalid dataname! Please check")

        return label_describe

    def Format_output_Description(self):

        format_describe = "[Format_output_Description]:\n" \
                        "eg: ```json\n [{\"entity\": \"Boeing E-3\", \"entity_type\": \"aircraft\", \"bounding_box\": [293, 21, 593, 449]}, {\"entity\": \"G36k\", \"entity_type\": \"weapon\", \"bounding_box\": [31, 94, 564, 612]}]\n```"
        
        return format_describe
