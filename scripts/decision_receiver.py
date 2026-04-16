#!/usr/bin/env python3
"""
Decision Receiver for External UI System

Receives decision messages from TerrainFormer via UDP and displays them.
Use this as a template for integrating with your UI system.

Usage:
    python decision_receiver.py --port 9999

On the inference side, run with --publish flag:
    python realtime_inference.py --mode folder --data-dir /path/to/data --publish
"""

import socket
import json
import argparse
import time
from datetime import datetime


# Action labels for display
ACTION_LABELS = {
    0: "STOP",
    1: "FORWARD_SLOW",
    2: "FORWARD_MEDIUM",
    3: "FORWARD_FAST",
    4: "BACKWARD_SLOW",
    5: "BACKWARD_MEDIUM",
    6: "BACKWARD_FAST",
    7: "TURN_LEFT_SHARP",
    8: "TURN_LEFT_MEDIUM",
    9: "TURN_LEFT_SLIGHT",
    10: "STRAIGHT",
    11: "TURN_RIGHT_SLIGHT",
    12: "TURN_RIGHT_MEDIUM",
    13: "TURN_RIGHT_SHARP",
    14: "FORWARD_LEFT",
    15: "FORWARD_RIGHT",
    16: "BACKWARD_LEFT",
    17: "BACKWARD_RIGHT",
}


class DecisionReceiver:
    """
    Receives and processes decision messages from TerrainFormer.

    Integrate this into your UI system by:
    1. Calling receive() in your main loop
    2. Processing the returned decision dict
    3. Updating your UI with action, confidence, etc.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 9999):
        self.host = host
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind((host, port))
        self.socket.setblocking(False)
        print(f"Decision receiver listening on {host}:{port}")

        # Statistics
        self.msg_count = 0
        self.last_msg_time = None

    def receive(self, timeout: float = 0.1) -> dict:
        """
        Receive a decision message (non-blocking).

        Returns:
            Decision dict or None if no message available
        """
        try:
            self.socket.settimeout(timeout)
            data, addr = self.socket.recvfrom(4096)
            message = json.loads(data.decode('utf-8'))

            self.msg_count += 1
            self.last_msg_time = time.time()

            return message
        except socket.timeout:
            return None
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
            return None
        except Exception as e:
            print(f"Receive error: {e}")
            return None

    def close(self):
        self.socket.close()


def format_decision(decision: dict) -> str:
    """Format decision for display."""
    if decision is None:
        return "No decision received"

    action = decision.get('action', -1)
    action_name = decision.get('action_name', ACTION_LABELS.get(action, 'Unknown'))
    confidence = decision.get('confidence', 0.0)
    inference_ms = decision.get('inference_ms', 0.0)

    # Build output string
    output = []
    output.append(f"╔══════════════════════════════════════╗")
    output.append(f"║  ACTION: {action_name:^26} ║")
    output.append(f"║  Confidence: {confidence*100:5.1f}%                  ║")
    output.append(f"║  Inference: {inference_ms:5.1f}ms                  ║")

    # Show top action probabilities
    if 'action_probs' in decision:
        output.append(f"╠══════════════════════════════════════╣")
        output.append(f"║  Top Actions:                        ║")
        for name, prob in decision['action_probs'].items():
            bar_len = int(prob * 20)
            bar = '█' * bar_len + '░' * (20 - bar_len)
            output.append(f"║  {name:12} {bar} {prob*100:4.1f}%║")

    # Show ground truth if available
    if 'gt_action_name' in decision:
        gt_name = decision['gt_action_name']
        correct = decision.get('correct', False)
        status = "✓ CORRECT" if correct else "✗ WRONG"
        output.append(f"╠══════════════════════════════════════╣")
        output.append(f"║  Ground Truth: {gt_name:^21} ║")
        output.append(f"║  Result: {status:^27} ║")

    output.append(f"╚══════════════════════════════════════╝")

    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description='Decision Receiver for UI')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                        help='Host to listen on')
    parser.add_argument('--port', type=int, default=9999,
                        help='UDP port to listen on')
    parser.add_argument('--json', action='store_true',
                        help='Output raw JSON instead of formatted')
    args = parser.parse_args()

    receiver = DecisionReceiver(host=args.host, port=args.port)

    print("\nWaiting for decisions from TerrainFormer...")
    print("Run inference with: python realtime_inference.py --publish\n")

    try:
        while True:
            decision = receiver.receive(timeout=0.5)

            if decision:
                # Clear screen for clean display
                print("\033[2J\033[H", end="")  # ANSI clear screen

                if args.json:
                    print(json.dumps(decision, indent=2))
                else:
                    print(format_decision(decision))
                    print(f"\nMessages received: {receiver.msg_count}")
                    print(f"Time: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        receiver.close()


if __name__ == '__main__':
    main()
