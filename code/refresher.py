import time
import pyautogui

def refresh_every_three_minutes():
    while True:
        pyautogui.press('f5')  # Simulates pressing the F5 key
        print("F5 key pressed. Waiting 3 minutes...")
        time.sleep(180)  # Wait for 3 minutes (180 seconds)

if __name__ == "__main__":
    refresh_every_three_minutes()
