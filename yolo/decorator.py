import openai

GENERATE_PYTHON_FUNCTION_TEMPLATE = """
"""

PREDICT_FUNCTION_SYSTEM_PROMPT = """What is the most likely output of the Python call
Return only a Python interpretation of the output, like a string, a list, a dictionary, etc.
No other explanation is needed.
"""
PREDICT_FUNCTION_OUTPUT_TEMPLATE = """
{function_name}({function_args}, **{function_kwargs})

The docstring for the function is:
{function_docstring}
"""

def yolo(func):
    # We don't run the function at all. We just use the function's name and docstring
    function_name = func.__name__
    function_docstring = func.__doc__

    def wrapper(*args, **kwargs):
        prompt = PREDICT_FUNCTION_OUTPUT_TEMPLATE.format(function_name=function_name,
                                                                  function_docstring=function_docstring,
                                                                  function_args=args, function_kwargs=kwargs, )
        messages = [
                {"role": "system", "content": PREDICT_FUNCTION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
        ]
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=messages,
            temperature=0,
            timeout=30,
        )

        out = response["choices"][0]["message"]["content"].strip()
        return out

    return wrapper


