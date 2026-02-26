# 📘 HELP — Guia Rápido

---

## 🟢 NVM — Node Version Manager

> Ferramenta para gerenciar versões do Node.js. Permite ter várias versões instaladas e alternar entre elas conforme necessário.

📖 **Instalação e documentação oficial:** [github.com/nvm-sh/nvm](https://github.com/nvm-sh/nvm)

### O que é Node.js?

**Node.js** é o ambiente que executa JavaScript fora do navegador.

---

### Comandos Essenciais

| Comando                 | O que faz                           |
| ----------------------- | ----------------------------------- |
| `nvm list`              | Lista as versões do Node instaladas |
| `nvm install 18.16.0`   | Instala a versão especificada       |
| `nvm use 18.16.0`       | Ativa a versão especificada         |
| `nvm current`           | Mostra a versão ativa no momento    |
| `nvm uninstall 18.16.0` | Remove a versão especificada        |
| `nvm root`              | Exibe a pasta raiz do NVM           |

---

### Fluxo comum de uso

```bash
# 1. Veja o que está instalado
nvm list

# 2. Instale a versão que o projeto precisa (se ainda não tiver)
nvm install 20.11.0

# 3. Ative essa versão
nvm use 20.11.0

# 4. Confirme
nvm current
```

### ⚠️ Dica

O `nvm use` vale apenas para a sessão atual do terminal. Para definir uma versão padrão permanente:

```bash
nvm alias default 18.16.0
```

---

## 🎨 Oh My Zsh — Terminal bonito como o de todo mundo

> Sabe aquele terminal colorido, cheio de ícones e com o nome da branch do Git aparecendo? É isso aqui.

📖 **Instalação e documentação oficial:** [ohmyz.sh/#install](https://ohmyz.sh/#install)

### Primeiro: entendendo o terminal

Quando você abre o terminal, existe um programa por trás interpretando seus comandos. Esse programa é chamado de **shell**.

Os dois mais comuns são:

| Shell    | Arquivo de configuração | Descrição                                               |
| -------- | ----------------------- | ------------------------------------------------------- |
| **Bash** | `~/.bashrc`             | O shell padrão da maioria dos sistemas Linux            |
| **Zsh**  | `~/.zshrc`              | Mais moderno, com recursos extras e mais personalizável |

O arquivo de configuração (`.bashrc` ou `.zshrc`) é lido toda vez que você abre um terminal. É nele que ficam variáveis de ambiente, aliases e configurações do seu shell.

---

### O que é Oh My Zsh?

**Oh My Zsh** é um framework para o Zsh que facilita a personalização do terminal. Com ele você instala temas e plugins com poucos comandos.

**Instalação:**

```bash
sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
```

Após instalar, o Oh My Zsh vai criar (ou sobrescrever) o arquivo `~/.zshrc` com as configurações dele.

---

### Personalizando o tema

Abra o arquivo de configuração:

```bash
nano ~/.zshrc
```

Procure a linha:

```bash
ZSH_THEME="robbyrussell"
```

Troque pelo tema que quiser. Um favorito popular:

```bash
ZSH_THEME="agnoster"
```

Para ver todos os temas disponíveis: [ohmyzsh/wiki/Themes](https://github.com/ohmyzsh/ohmyzsh/wiki/Themes)

Depois de salvar, aplique as mudanças:

```bash
source ~/.zshrc
```

---

### Plugins úteis

No `~/.zshrc`, procure a linha `plugins=(git)` e adicione os que quiser:

```bash
plugins=(git zsh-autosuggestions zsh-syntax-highlighting)
```

| Plugin                    | O que faz                                                |
| ------------------------- | -------------------------------------------------------- |
| `git`                     | Atalhos e info de branch no terminal (já vem por padrão) |
| `zsh-autosuggestions`     | Sugere comandos enquanto você digita                     |
| `zsh-syntax-highlighting` | Colore o comando enquanto você escreve                   |

> Alguns plugins precisam ser instalados separadamente. Consulte a documentação de cada um.
