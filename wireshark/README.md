# Wireshark plugin

The Lua plugin sends the open capture to the local detector service as one batch and adds the returned flow result to every packet that belongs to that flow.
The current model is flow-based, so packets are evidence for a scored flow rather than independently classified samples.

## Start the scoring server

Build and run the deployment image from the repository root:

```sh
./deploy.sh
docker run --rm -e PORT=8080 -p 127.0.0.1:8080:8080 network-anomaly-detector:local
```

Keep the port bound to `127.0.0.1` unless you add authentication and transport security in front of the service.
Packet captures can contain credentials and other sensitive data.

Verify the server before opening Wireshark:

```sh
curl --fail http://127.0.0.1:8080/health
```

## Install the plugin

Open **Help > About Wireshark > Folders** and locate the Personal Lua Plugins folder.
Create that folder if it does not exist, then copy `ai_anomaly_detector.lua` into it.
On macOS, the folder is commonly `~/.local/lib/wireshark/plugins` for current Wireshark releases.

Restart Wireshark or select **Analyze > Reload Lua Plugins**.
Open a PCAP or PCAPNG file and select **Tools > AI Network Anomaly > Analyze Open Capture**.

The default settings call `http://127.0.0.1:8080/api/v1/captures/score` with `/usr/bin/curl`.
Change them under **Preferences > Protocols > AI Network Anomaly** if needed.

## Use the annotations

Flagged packets include an **AI Network Anomaly** section in Packet Details and an `[AI: ...]` suffix in the Info column.
The plugin adds these display-filter fields:

- `ai_anomaly.score`
- `ai_anomaly.is_anomaly`
- `ai_anomaly.prediction`
- `ai_anomaly.confidence`
- `ai_anomaly.flow_id`

Use this filter to show packets in anomalous flows:

```text
ai_anomaly.is_anomaly == 1
```

You can add `ai_anomaly.score` or `ai_anomaly.prediction` as a custom packet-list column from the field context menu.

## Current boundaries

- Analysis is explicit and batch-oriented because Wireshark's Lua API does not provide a portable asynchronous HTTP client.
- The Wireshark window waits while the local server extracts and scores the flows.
- The plugin currently supports macOS and Linux because it invokes `curl` through a POSIX shell.
- A high anomaly score means unusual behavior, not proof of an attack.
- The legacy deployment model reports `Unknown anomaly` or `Benign`.
- A model bundle trained by the Colab pipeline can additionally provide covered attack-family predictions once the deployment runtime is switched to that bundle.
- XSS cannot be confirmed from encrypted flow metadata alone.
