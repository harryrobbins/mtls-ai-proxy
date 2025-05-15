#
# def get_openai_mtls_client(asynchronous=False):
#     ctx = ssl.create_default_context(cafile=CA_PATH)
#     ctx.load_cert_chain(certfile=CERTIFICATE_PATH, keyfile=KEY_PATH)
#
#     if asynchronous:
#         httpx_client = httpx.AsyncClient(verify=ctx)
#         openai_client = openai.AsyncOpenAI(http_client=httpx_client,
#                                            base_url="https://localhost:8443",
#                                            api_key=LITELLM_PROXY_KEY)
#     else:
#         httpx_client = httpx.Client(verify=ctx)
#         openai.OpenAI(http_client=httpx_client,
# #                                       base_url="https://localhost:8443",
# #                                       api_key=LITELLM_PROXY_KEY)
#     return openai_client
#
#
# client = get_openai_mtls_client()
# LITELLM_PROXY_MODEL = "ollama-qwen3"
#
# response = client.chat.completions.create(
#     model=LITELLM_PROXY_MODEL,
#     messages=[
#         {
#             "role": "user",
#             "content": "Write a one-sentence bedtime story about a unicorn."
#         },
#     ],
#     max_tokens=5
# )
import openai
client = openai.OpenAI( base_url="https://localhost:8443",
                                      api_key=LITELLM_PROXY_KEY)
# response = client.chat.completions.create(
#     model=LITELLM_PROXY_MODEL,
#     messages=[
#         {
#             "role": "user",
#             "content": "Write a one-sentence bedtime story about a unicorn."
#         },
#     ],
#     max_tokens=5