import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scapy.all import IP, TCP, UDP, Ether, Raw, wrpcap

from evidence import FlowKey, collect_evidence

REPO_ROOT = Path(__file__).resolve().parent.parent


def _http_capture(path: Path) -> None:
    """One plaintext HTTP conversation plus one unrelated UDP flow."""

    client, server = "10.0.0.5", "93.184.216.34"
    request = (
        b"GET /admin/login.php HTTP/1.1\r\n"
        b"Host: example.com\r\n"
        b"User-Agent: curl/8.4.0\r\n\r\n"
    )
    response = (
        b"HTTP/1.1 200 OK\r\n"
        b"Content-Type: text/html\r\n"
        b"Content-Length: 12\r\n\r\n"
        b"<html>hi</h>"
    )
    packets = [
        Ether() / IP(src=client, dst=server) / TCP(sport=51000, dport=80, flags="S", seq=0),
        Ether() / IP(src=server, dst=client) / TCP(sport=80, dport=51000, flags="SA", seq=0, ack=1),
        Ether() / IP(src=client, dst=server) / TCP(sport=51000, dport=80, flags="A", seq=1, ack=1),
        Ether() / IP(src=client, dst=server) / TCP(sport=51000, dport=80, flags="PA", seq=1, ack=1) / Raw(request),
        Ether() / IP(src=server, dst=client) / TCP(sport=80, dport=51000, flags="PA", seq=1, ack=len(request) + 1) / Raw(response),
        Ether() / IP(src=client, dst=server) / UDP(sport=5353, dport=53) / Raw(b"\x00\x01"),
    ]
    for index, packet in enumerate(packets):
        packet.time = 1_700_000_000.0 + index * 0.01
    wrpcap(str(path), packets)


class FlowKeyTests(unittest.TestCase):
    def test_both_directions_of_a_conversation_share_one_key(self):
        forward = FlowKey.build("10.0.0.5", 51000, "93.184.216.34", 80, 6)
        backward = FlowKey.build("93.184.216.34", 80, "10.0.0.5", 51000, 6)

        self.assertEqual(forward, backward)

    def test_different_conversations_do_not_collide(self):
        first = FlowKey.build("10.0.0.5", 51000, "93.184.216.34", 80, 6)
        second = FlowKey.build("10.0.0.5", 51001, "93.184.216.34", 80, 6)

        self.assertNotEqual(first, second)

    def test_parses_its_own_text_form(self):
        key = FlowKey.build("10.0.0.5", 51000, "93.184.216.34", 80, 6)

        self.assertEqual(FlowKey.parse(key.as_text()), key)


class CollectEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.pcap = Path(self.temporary.name) / "http.pcap"
        _http_capture(self.pcap)
        self.addCleanup(self.temporary.cleanup)
        self.http_key = FlowKey.build("10.0.0.5", 51000, "93.184.216.34", 80, 6)

    def test_returns_every_flow_when_no_keys_are_requested(self):
        evidence = collect_evidence(self.pcap)

        self.assertEqual(len(evidence), 2)
        self.assertIn(self.http_key, evidence)

    def test_returns_only_the_requested_flow(self):
        evidence = collect_evidence(self.pcap, [self.http_key])

        self.assertEqual(list(evidence), [self.http_key])

    def test_packet_indices_point_back_into_the_capture(self):
        evidence = collect_evidence(self.pcap, [self.http_key])

        self.assertEqual(evidence[self.http_key].packet_indices, [0, 1, 2, 3, 4])
        self.assertEqual(evidence[self.http_key].packet_count, 5)

    def test_both_directions_are_gathered_under_one_stream(self):
        evidence = collect_evidence(self.pcap, [self.http_key])
        udp_key = FlowKey.build("10.0.0.5", 5353, "10.0.0.5", 53, 17)

        self.assertIsNotNone(evidence[self.http_key].stream_id)
        self.assertNotIn(udp_key, evidence)

    def test_records_the_observation_interval(self):
        evidence = collect_evidence(self.pcap, [self.http_key])
        found = evidence[self.http_key]

        self.assertAlmostEqual(found.first_seen, 1_700_000_000.0, places=3)
        self.assertAlmostEqual(found.last_seen, 1_700_000_000.04, places=3)

    def test_recovers_the_plaintext_http_request_line_and_headers(self):
        evidence = collect_evidence(self.pcap, [self.http_key])
        request = evidence[self.http_key].http_requests[0]

        self.assertEqual(request["method"], "GET")
        self.assertEqual(request["path"], "/admin/login.php")
        self.assertEqual(request["host"], "example.com")
        self.assertEqual(request["user_agent"], "curl/8.4.0")
        self.assertEqual(request["packet_index"], 3)

    def test_records_response_status_without_retaining_the_body(self):
        evidence = collect_evidence(self.pcap, [self.http_key])
        response = evidence[self.http_key].http_responses[0]

        self.assertEqual(response["status"], 200)
        self.assertEqual(response["content_type"], "text/html")
        self.assertEqual(response["content_length"], 12)
        serialized = json.dumps(evidence[self.http_key].to_dict())
        self.assertNotIn("<html>hi", serialized)

    def test_bounds_how_many_packet_indices_are_kept(self):
        evidence = collect_evidence(self.pcap, [self.http_key], max_packets_per_flow=2)
        found = evidence[self.http_key]

        self.assertEqual(len(found.packet_indices), 2)
        self.assertEqual(found.packet_count, 5)

    def test_a_flow_absent_from_the_capture_yields_no_entry(self):
        absent = FlowKey.build("192.0.2.1", 1234, "192.0.2.2", 443, 6)

        evidence = collect_evidence(self.pcap, [absent])

        self.assertEqual(evidence, {})

    def test_evidence_is_json_serializable(self):
        evidence = collect_evidence(self.pcap, [self.http_key])

        json.dumps({key.as_text(): value.to_dict() for key, value in evidence.items()})


class RepositoryFixtureTests(unittest.TestCase):
    """A real capture that contains no HTTP at all must still read cleanly."""

    def test_reads_the_dhcp_fixture_without_http(self):
        evidence = collect_evidence(REPO_ROOT / "deploy" / "dhcp.pcap")

        self.assertGreaterEqual(len(evidence), 1)
        for found in evidence.values():
            self.assertEqual(found.http_requests, [])


class CommandLineTests(unittest.TestCase):
    def test_prints_json_for_a_capture(self):
        with TemporaryDirectory() as temporary:
            pcap = Path(temporary) / "http.pcap"
            _http_capture(pcap)

            completed = subprocess.run(
                [sys.executable, "-m", "evidence", str(pcap)],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                check=True,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(len(payload), 2)

    def test_filters_to_one_flow(self):
        with TemporaryDirectory() as temporary:
            pcap = Path(temporary) / "http.pcap"
            _http_capture(pcap)

            completed = subprocess.run(
                [
                    sys.executable, "-m", "evidence", str(pcap),
                    "--flow", "10.0.0.5:51000-93.184.216.34:80/6",
                ],
                capture_output=True,
                text=True,
                cwd=REPO_ROOT,
                check=True,
            )

        payload = json.loads(completed.stdout)
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["http_requests"][0]["method"], "GET")


if __name__ == "__main__":
    unittest.main()
