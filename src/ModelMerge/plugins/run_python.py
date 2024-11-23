import os
import asyncio
import logging
import tempfile
import ast
import builtins

def get_dangerous_attributes(node):
    # 简单的代码审查，检查是否包含某些危险关键词
    dangerous_keywords = ['os', 'subprocess', 'sys', 'import', 'eval', 'exec', 'open']
    if isinstance(node, ast.Name):
        return node.id in dangerous_keywords
    elif isinstance(node, ast.Attribute):
        return node.attr in dangerous_keywords
    return False

def check_code_safety(code):
    try:
        # 解析代码为 AST
        tree = ast.parse(code)

        # 检查所有节点
        for node in ast.walk(tree):
            # 检查危险属性访问
            if get_dangerous_attributes(node):
                return False

            # 检查危险的调用
            if isinstance(node, ast.Call):
                if isinstance(node.func, (ast.Name, ast.Attribute)):
                    if get_dangerous_attributes(node.func):
                        return False

            # 检查字符串编码/解码操作
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in ('encode', 'decode'):
                    return False

        return True
    except SyntaxError:
        return False

async def run_python_script(code, timeout=10):
    if not check_code_safety(code):
        return "Code contains potentially dangerous operations.\n\n"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
        temp_file.write(code)
        temp_file_name = temp_file.name

    try:
        process = await asyncio.create_subprocess_exec(
            'python', temp_file_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
            stdout = stdout.decode()
            stderr = stderr.decode()
            return_code = process.returncode
        except asyncio.TimeoutError:
            # 使用 SIGTERM 信号终止进程
            process.terminate()
            await asyncio.sleep(0.1)  # 给进程一点时间来终止
            if process.returncode is None:
                # 如果进程还没有终止，使用 SIGKILL
                process.kill()
            return "Process execution timed out."

        mess = (
            f"Execution result:\n{stdout}\n",
            f"Stderr:\n{stderr}\n" if stderr else "",
            f"Return Code: {return_code}\n" if return_code else "",
        )
        mess = "".join(mess)
        return mess

    except Exception as e:
        logging.error(f"Error executing code: {str(e)}")
        return f"Error: {str(e)}"

    finally:
        try:
            os.unlink(temp_file_name)
        except Exception as e:
            logging.error(f"Error deleting temporary file: {str(e)}")

# 使用示例
async def main():
    code = """
print("Hello, World!")
"""
    code = """
def add(a, b):
    return a + b

result = add(5, 3)
print(result)
    """
    result = await run_python_script(code)
    print(result)

if __name__ == "__main__":
    asyncio.run(main())