import inspect
from typing import Any, Callable, Dict, List, Union, Optional
from pydantic import BaseModel, Field


class Parameter(BaseModel):
    class Config:
        arbitrary_types_allowed = True

    name: str = Field(description="The name of the parameter")
    description: Optional[str] = Field(
        default=None, description="The description of the parameter"
    )
    type: str = Field(description="The type of the parameter")
    default: Any = Field(description="The default value of the parameter")
    required: bool = Field(description="Whether the parameter is required")

    def to_openai(self):
        return {
            self.name: {
                "description": self.description,
                "type": self.type,
            }
        }

class FunctionSchema:
    """Class that consumes a function and can return a schema required by
    different LLMs for function calling.
    """

    name: str = Field(description="The name of the function")
    description: str = Field(description="The description of the function")
    signature: str = Field(description="The signature of the function")
    output: str = Field(description="The output of the function")
    parameters: List[Parameter] = Field(description="The parameters of the function")

    def __init__(self, function: Union[Callable, BaseModel]):
        self.function = function
        if callable(function):
            self._process_function(function)
        elif isinstance(function, BaseModel):
            raise NotImplementedError("Pydantic BaseModel not implemented yet.")
        else:
            raise TypeError("Function must be a Callable or BaseModel")

    def _process_function(self, function: Callable):
        self.name = function.__name__
        self.description = str(inspect.getdoc(function))
        self.signature = str(inspect.signature(function))
        self.output = str(inspect.signature(function).return_annotation)
        parameters = []
        for param in inspect.signature(function).parameters.values():
            parameters.append(
                Parameter(
                    name=param.name,
                    type=param.annotation.__name__,
                    default=param.default,
                    required=param.default is inspect.Parameter.empty,
                )
            )
        self.parameters = parameters

    def to_openai(self):
        schema_dict = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        param.name: {
                            "description": (
                                param.description
                                if isinstance(param.description, str)
                                else "None provided"
                            ),
                            "type": self._openai_type_mapping(param.type),
                        }
                        for param in self.parameters
                    },
                    "required": [
                        param.name for param in self.parameters if param.required
                    ],
                },
            },
        }
        return schema_dict

    def _openai_type_mapping(self, param_type: str) -> str:
        if param_type == "int":
            return "number"
        elif param_type == "float":
            return "number"
        elif param_type == "str":
            return "string"
        elif param_type == "bool":
            return "boolean"
        else:
            return "object"


def get_schema_pydantic(model: BaseModel) -> Dict[str, Any]:
    signature_parts = []
    for field_name, field_model in model.__annotations__.items():
        field_info = model.__fields__[field_name]
        default_value = field_info.default

        if default_value:
            default_repr = repr(default_value)
            signature_part = (
                f"{field_name}: {field_model.__name__} = {default_repr}"
            )
        else:
            signature_part = f"{field_name}: {field_model.__name__}"

        signature_parts.append(signature_part)
    signature = f"({', '.join(signature_parts)}) -> str"
    schema = FunctionSchema(
        name=model.__class__.__name__,
        description=model.__doc__,
        signature=signature,
        output="",  # TODO: Implement output
    )
    return schema
