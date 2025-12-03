### Mind2Web
_MIND2WEB_SYSTEM = """You are an assistant trained to navigate the web. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
1. `CLICK`: Click on an element, value is the element to click and the position [x,y] is required.
2. `TYPE`: Type a string into an element, value is the string to type and the position [x,y] is required.
3. `SELECT`: Select a value for an element, value is the value to select and the position [x,y] is required.

Format the action as a dictionary with the following keys:
{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

_MIND2WEB_SYSTEM_THINKING = """You are an assistant trained to navigate the web. 
Given a task instruction, a screenshot, and a last history action summary, output the think and ext action and wait for the next observation. 
The think must strictly follow these reasoning steps:
(1) Progress Estimation: Task Decomposition, Planning and Progress Estimation
(2) Decision Making: Strategy Formulation
(3) History Summary: Update the history summary according the action you executed

## Action Space
1. `CLICK`: Click on an element, value is the element to click and the position [x,y] is required.
2. `TYPE`: Type a string into an element, value is the string to type and the position [x,y] is required.
3. `SELECT`: Select a value for an element, value is the value to select and the position [x,y] is required.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.

## Output Format
<think>
<Progress Estimation>
...
</Progress Estimation>
<Decision Reasoning>
...
</Decision Reasoning>
<action>
{{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}}
</action>
<History Summary>
...
</History Summary>
<think>

If value or position is not applicable, set it as `None`.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1, where (x=0, y=0) denotes the top-left corner of the screen, 
with x increasing rightward and y increasing downward.
"""

_MIND2WEB_USER = """{system}
Task: {task}
Observation: <|image_1|>
Action History: {action_history}
What is the next action?
"""

def mind2web_to_qwen(task, action_history, answer_dict=None, think=True, memory=None):
    transformed_data = []
    user_content = []

    if think:
        system_prompt = _MIND2WEB_SYSTEM_THINKING
    else:
        system_prompt = _MIND2WEB_SYSTEM

    user_content.append({"type": "text", "text": system_prompt})
    user_content.append({"type": "text", "text": f"Task: {task}"})
    user_content.extend(action_history)
    if memory is not None:
        user_content.append({"type": "text", "text": f"History summary: {memory}\n"})
    transformed_data.append(
                {
                    "role": "user",
                    "content": user_content,
                },
            )
    return transformed_data