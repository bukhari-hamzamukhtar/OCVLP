"""Prompt for object-centric symbol extraction.

It is VLP's own get_objects prompt (method/DSL/dsl_with_img_repr.py) with two
changes, and nothing else:
  1. every object gets a bounding box, in the JSON form Qwen2.5-VL was trained
     to produce for grounding ({"bbox_2d": [...], "label": ...});
  2. every object is listed separately, because counts and relations need all
     of them.
"""

OC_MAX_NEW_TOKENS = 700

OC_PROMPT = """
        ## Task
        List every object in the image. Describe each object using only the provided lists, and give its bounding box.

        **Objects:** {objects}
        **Properties:** {properties}

        ## Rules
        1. Only use objects/properties from the provided lists
        2. List every object separately, even when two objects look alike
        3. Give each bounding box as [x1, y1, x2, y2] in pixel coordinates of the image
        4. Return an empty list if no valid objects are found
        5. No explanations or additional text

        ## Output Format
        ```json
        [
            {{"label": "object_name", "properties": ["property1", "property2"], "bbox_2d": [x1, y1, x2, y2]}},
            {{"label": "object_name", "properties": ["property1"], "bbox_2d": [x1, y1, x2, y2]}}
        ]
        ```

        **If no valid objects:** `[]`

        **Analyze the image now:**
        """


def oc_prompt(objects, properties):
    return OC_PROMPT.format(objects=objects, properties=properties)
