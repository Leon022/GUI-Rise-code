import pdb
### AITW
_AITW_SYSTEM = """You are an assistant trained to navigate the mobile phone. 
Given a task instruction, a screen observation, and an action history sequence, 
output the next action and wait for the next observation. 
Here is the action space:
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
2. `TYPE`: Type a string into an element, value is a string to type and the position is not applicable.
3. `SELECT`: Select a value for an element, value is the value to select and the position is not applicable.
4. `SCROLL UP`: Scroll up for the screen.
5. `SCROLL DOWN`: Scroll down for the screen.
6. `SCROLL LEFT`: Scroll left for the screen.
7. `SCROLL RIGHT`: Scroll right for the screen.
8. `PRESS BACK`: Press for returning to the previous step, value and position are not applicable.
9. `PRESS HOME`: Press for returning to the home screen, value and position are not applicable.
10. `PRESS ENTER`: Press for submitting the input content, value and position are not applicable.
11. `STATUS TASK COMPLETE`: Indicate the task is completed, value and position are not applicable.
12. `STATUS TASK IMPOSSIBLE `: Indicate the task is impossible to complete, value and position are not applicable.
"""

_AITW_SYSTEM_THINKING = """You are an assistant trained to navigate the mobile. 
Given a task instruction, a screenshot, and a last history action summary, output the think and action and wait for the next observation. 
The think must strictly follow these reasoning steps:
(1) Progress Estimation: Task Decomposition, Planning and Progress Estimation
(2) Decision Reasoning: Strategy Formulation
(3) Action Prediction: Giving the answer action in <action>...</action>
(4) History Summary: Update the history summary according the action you executed

## Action Space
1. `CLICK`: Click on an element, value is not applicable and the position [x,y] is required. 
2. `INPUT`: Input a string into an element, value is a string to type and the position is not applicable.
3. `REMEMBER`: Remember a string, value is a string to type and the position is not applicable.
4. `SCROLL UP`: Scroll up for the screen.
5. `SCROLL DOWN`: Scroll down for the screen.
6. `SCROLL LEFT`: Scroll left for the screen.
7. `SCROLL RIGHT`: Scroll right for the screen.
8. `PRESS BACK`: Press for returning to the previous step, value and position are not applicable.
9. `PRESS HOME`: Press for returning to the home screen, value and position are not applicable.
10. `PRESS ENTER`: Press for submitting the input content, value and position are not applicable.
11. `STATUS TASK COMPLETE`: Indicate the task is completed, value and position are not applicable.
12. `STATUS TASK IMPOSSIBLE `: Indicate the task is impossible to complete, value and position are not applicable.

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

_AITW_SYSTEM_json = """
Format the action as a dictionary with the following keys:
{'action': 'ACTION_TYPE', 'value': 'element', 'position': [x,y]}

If value or position is not applicable, set it as `None`.
Position represents the relative coordinates on the screenshot and should be scaled to a range of 0-1.
"""

_AITW_USER = """{system}
Task: {task}
Action History: {action_history}
Observation: <|image_1|>
"""

def aitw_to_qwen(task, action_history, answer_dict=None, skip_readme=False, think=True, memory=None):
    transformed_data = []
    user_content = []

    if think:
        system_prompt = _AITW_SYSTEM_THINKING
    elif not skip_readme: 
        system_prompt = _AITW_SYSTEM + _AITW_SYSTEM_json
    else:
        system_prompt = _AITW_SYSTEM_json

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