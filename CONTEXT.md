# Domain glossary

## Packet

A single captured network frame with a timestamp and protocol metadata. A packet
can support a finding, but most attack families cannot be identified reliably
from one packet alone.

## Flow

A bidirectional conversation grouped by source, destination, ports, protocol,
and time. A flow summarizes the packets exchanged by two endpoints.

## Observation window

A bounded time interval containing flow and packet measurements. Windows make
rate changes, fan-out, repeated attempts, and service degradation observable.

## Application signal

An observation emitted by a monitored application, such as an authentication
failure, HTTP status, request latency, or validated web-attack event. It is not
inferred from encrypted packet contents.

## Detector signal

A scored claim produced by one detector about one flow or observation window.
It records the detector, score, threshold, candidate attack family, and the
measurements that caused the score.

## Attack family

A stable category such as DDoS, brute force, or web injection. A family is
reported only when the active detectors have declared coverage for it.

## Evidence

The immutable observations and measurements that support a detector signal or
finding. Evidence may reference packet indices, flows, windows, or application
signals without retaining packet payloads.

## Finding

A user-facing, evidence-backed explanation of suspicious behavior. A finding
states what was observed, the affected target and interval, its confidence and
severity, and recommended investigation steps.

## Incident

One or more related findings correlated by target, attack family, and time.
An incident is the unit a user acknowledges or resolves.

## Confidence

How strongly the available evidence supports the stated attack family.
Confidence is distinct from severity and from a raw anomaly score.

## Severity

The estimated operational impact of a finding, based on magnitude, duration,
affected assets, and application health. Severity does not express model
certainty.

## Coverage

The attack families and evidence sources a detector has been validated to
recognize. Missing telemetry reduces coverage and must be visible to the user.
