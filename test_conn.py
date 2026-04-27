import asyncio
import ollama

async def test():
    client = ollama.AsyncClient()
    try:
        models = await client.list()
        print("Conexión exitosa. Modelos encontrados:", len(models['models']))
    except Exception as e:
        print("Error de conexión:", str(e))

asyncio.run(test())
