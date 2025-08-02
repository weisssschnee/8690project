# core/strategy/llm_trader_adapters.py
import logging
import json
from abc import ABC, abstractmethod
from typing import Dict, Any
import os

# --- 修复 Gemini 导入检测 ---
GEMINI_AVAILABLE = False
genai = None
GenerationConfig = None
Tool = None
FunctionDeclaration = None
Part = None

try:
    import google.generativeai as genai

    print("✅ Step 1: google.generativeai 导入成功")

    try:
        from google.generativeai.types import GenerationConfig

        print("✅ Step 2: GenerationConfig 导入成功")
    except ImportError as e:
        print(f"⚠️ Step 2: GenerationConfig 导入失败: {e}")
        # 尝试直接从 genai 获取
        GenerationConfig = getattr(genai, 'GenerationConfig', None)
        if GenerationConfig:
            print("✅ Step 2b: 从 genai 获取 GenerationConfig 成功")

    try:
        from google.generativeai.types import Tool, FunctionDeclaration

        print("✅ Step 3: Tool, FunctionDeclaration 导入成功")
    except ImportError as e:
        print(f"⚠️ Step 3: Tool, FunctionDeclaration 导入失败: {e}")
        # 尝试从 genai 获取
        Tool = getattr(genai, 'Tool', None)
        FunctionDeclaration = getattr(genai, 'FunctionDeclaration', None)
        if Tool and FunctionDeclaration:
            print("✅ Step 3b: 从 genai 获取 Tool, FunctionDeclaration 成功")

    try:
        from google.generativeai.types import Part

        print("✅ Step 4: Part 导入成功")
    except ImportError as e:
        print(f"⚠️ Step 4: Part 导入失败: {e}")
        # 尝试从 genai 获取
        Part = getattr(genai, 'Part', None)
        if Part:
            print("✅ Step 4b: 从 genai 获取 Part 成功")
        else:
            # 创建一个简单的 Part 类
            class Part:
                @staticmethod
                def from_function_response(name, response):
                    # 创建一个简单的对象来传递给模型
                    return {"name": name, "response": response}


            print("✅ Step 4c: 创建简单的 Part 类成功")

    # 如果基本的 genai 可用，就认为 Gemini 可用
    if genai is not None:
        GEMINI_AVAILABLE = True
        print("✅ Gemini 最终检测: 可用")
        logging.info("✅ Gemini library successfully imported")
    else:
        print("❌ Gemini 最终检测: 不可用 (genai is None)")

except ImportError as e:
        print(f"❌ Step 1: google.generativeai 导入失败: {e}")

# --- 修复 DeepSeek 导入检测 ---
try:
    from openai import OpenAI

    DEEPSEEK_AVAILABLE = True
    DEEPSEEK_CLIENT_TYPE = "openai"
    print("✅ DeepSeek 库检测: 使用 OpenAI 兼容客户端")
    logging.info("✅ DeepSeek library available using OpenAI client")
except ImportError as e:
    print(f"❌ DeepSeek 库检测失败: {e}")
    DEEPSEEK_AVAILABLE = False
    DEEPSEEK_CLIENT_TYPE = None
    logging.warning(f"OpenAI SDK not found for DeepSeek. Error: {e}")

logger = logging.getLogger(__name__)


def web_search(query: str) -> str:
    logger.info(f"[LLM Tool] Executing web_search with query: '{query}'")
    if not query:
        logger.warning("[LLM Tool] web_search called with an empty or None query.")
        return "Search failed: The search query was empty. Please provide a specific query."
    query = query.lower()
    if "earnings report" in query or "财报" in query:
        return "模拟搜索结果：[最近的财报显示，该公司的收入同比增长了15%，超出了市场预期。但管理层对下一季度的指引持谨慎态度。]"
    elif "product launch" in query or "新产品" in query:
        return "模拟搜索结果：[该公司最近发布了一款备受好评的新产品，分析师认为这可能在未来几个季度大幅提升其市场份额。]"
    else:
        return "模拟搜索结果：[关于'{}'没有找到特别重要的新闻。市场情绪中性。]".format(query)


class BaseLLMTraderAdapter(ABC):
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("API key is required.")
        self.api_key = api_key
        self.model_name = model_name

    @abstractmethod
    def get_decision(self, prompt_context: str) -> Dict[str, Any]:
        pass

    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        try:
            start_index = response_text.find('{')
            end_index = response_text.rfind('}') + 1
            if start_index == -1 or end_index == 0:
                raise json.JSONDecodeError("No JSON object found in response.", response_text, 0)

            json_str = response_text[start_index:end_index]
            parsed_json = json.loads(json_str)

            if not all(k in parsed_json for k in ["decision", "confidence", "reasoning"]):
                raise ValueError("Parsed JSON is missing required keys.")
            if parsed_json["decision"].upper() not in ["BUY", "SELL", "HOLD"]:
                raise ValueError(f"Invalid 'decision' value: {parsed_json['decision']}.")

            return parsed_json
        except (json.JSONDecodeError, ValueError) as e:
            return {"error": f"Failed to parse LLM response: {e}. Response: {response_text}"}
        except Exception as e:
            return {"error": f"Unexpected error during response parsing: {e}"}


class GeminiTraderAdapter(BaseLLMTraderAdapter):
    def __init__(self, api_key: str, model_name: str):
        if not GEMINI_AVAILABLE:
            raise ImportError("Gemini library not available.")
        super().__init__(api_key, model_name)

        # vvvvvvvvvvvvvvvvvvvv START OF DEFINITIVE FIX vvvvvvvvvvvvvvvvvvvv

        # --- Step 1: Prepare configuration options in a dictionary ---
        config_options = {
            "api_key": self.api_key
        }

        # --- Step 2: Check for proxy and add transport options if found ---
        proxy_url = os.environ.get('HTTPS_PROXY') or os.environ.get('HTTP_PROXY')
        if proxy_url:
            logger.critical(
                f"Gemini Fix: Proxy detected ({proxy_url}). Forcing REST transport to ensure proxy is used.")
            config_options['transport'] = 'rest'
            config_options['client_options'] = {"api_endpoint": "generativelanguage.googleapis.com"}
        else:
            logger.warning("Gemini Fix: No proxy found in environment. Regional errors may occur.")

        # --- Step 3: Make a SINGLE, definitive call to genai.configure ---
        print(f"🔧 Configuring Gemini API with options: {config_options}")
        genai.configure(**config_options)

        # ^^^^^^^^^^^^^^^^^^^^ END OF DEFINITIVE FIX ^^^^^^^^^^^^^^^^^^^^

        # The rest of the initialization remains the same
        self.tools = Tool(function_declarations=[
            FunctionDeclaration(
                name="web_search",
                description="Search for the latest, most specific information about a company, like earnings, products, or competitors.",
                parameters={"type": "OBJECT", "properties": {"query": {"type": "STRING"}}, "required": ["query"]},
            )
        ])
        self.model = genai.GenerativeModel(model_name=self.model_name, tools=[self.tools])
        print(f"✅ Gemini 适配器初始化成功")

    def get_decision(self, prompt_context: str) -> Dict[str, Any]:
        logger.info(f"Sending request to Gemini model: {self.model_name}")
        try:
            chat = self.model.start_chat()
            response = chat.send_message(
                prompt_context,
                generation_config=GenerationConfig(temperature=0.2)
            )

            # 检查是否有函数调用
            if (response.candidates and
                    response.candidates[0].content and
                    response.candidates[0].content.parts and
                    hasattr(response.candidates[0].content.parts[0], 'function_call') and
                    response.candidates[0].content.parts[0].function_call):

                function_call = response.candidates[0].content.parts[0].function_call
                if function_call.name == "web_search":
                    query = function_call.args["query"]
                    search_result = web_search(query)

                    response = chat.send_message(
                        Part.from_function_response(
                            name="web_search",
                            response={"result": search_result},
                        ),
                    )

            return self._parse_llm_response(response.text)

        except Exception as e:
            logger.error(f"An error occurred while calling Gemini API: {e}", exc_info=True)
            return {"error": f"Gemini API call failed: {e}"}


class DeepSeekTraderAdapter(BaseLLMTraderAdapter):
    """与 DeepSeek API 通信的适配器（使用 OpenAI 兼容客户端）。"""

    def __init__(self, api_key: str, model_name: str ):
        if not DEEPSEEK_AVAILABLE:
            raise ImportError("DeepSeek library is not available.")
        super().__init__(api_key, model_name)

        print(f"🔧 正在配置 DeepSeek (OpenAI兼容) 客户端...")
        # 使用 OpenAI 兼容客户端
        from openai import OpenAI
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"  # DeepSeek API 端点
        )
        print(f"✅ DeepSeek 适配器初始化成功")

    def get_decision(self, prompt_context: str) -> Dict[str, Any]:
        logger.info(f"Sending request to DeepSeek model: {self.model_name}")
        try:
            # FIX 2: The tool list ONLY contains web_search
            tools = [{"type": "function", "function": {
                "name": "web_search",
                "description": "Search for the latest, most specific information about a company.",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
            }}]

            messages = [{"role": "user", "content": prompt_context}]

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2
            )

            response_message = response.choices[0].message

            if hasattr(response_message, 'tool_calls') and response_message.tool_calls:
                # FIX 3: This block correctly handles the DeepSeek API format AND only handles web_search
                assistant_message_for_history = {
                    "role": response_message.role,
                    "tool_calls": [{"id": tc.id, "type": tc.type,
                                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}} for tc
                                   in response_message.tool_calls],
                }
                messages.append(assistant_message_for_history)

                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    if function_name == "web_search":
                        function_args = json.loads(tool_call.function.arguments)
                        function_response_content = web_search(query=function_args.get("query"))
                        messages.append({
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": function_response_content,
                        })

                second_response = self.client.chat.completions.create(model=self.model_name, messages=messages)
                response_message = second_response.choices[0].message
                # ^^^^^^^^^^^^^^^^^^^^ END OF DEFINITIVE FIX ^^^^^^^^^^^^^^^^^^^^

            # --- Parse the final text reply ---
            return self._parse_llm_response(response_message.content)

        except Exception as e:

            logger.error(f"An error occurred while calling DeepSeek API: {e}", exc_info=True)
            return {"error": f"DeepSeek API call failed: {e}"}