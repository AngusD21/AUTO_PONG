
import sys, socket

HOST = "127.0.0.1"
PORT = 8765

def main():
    if len(sys.argv) < 2:
        print("Usage: python send_cmd.py \"T,120,200,0\"")
        return
    cmd = sys.argv[1]
    if not cmd.endswith("\n"):
        cmd += "\n"
    with socket.create_connection((HOST, PORT), timeout=2.0) as s:
        # read greeting
        try:
            _ = s.recv(1024)
        except Exception:
            pass
        s.sendall(cmd.encode("utf-8"))
        try:
            resp = s.recv(1024).decode("utf-8").strip()
            if resp:
                print(resp)
        except Exception:
            pass

if __name__ == "__main__":
    main()
