# IAtest

Para configurar o ambiente:
```
conda create -y -n iatests python
```

Instalando pacotes:

```
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Para testes com o OpenHermes:

```
pip install ctransformers[cuda] transformers[torch]
```
