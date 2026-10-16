-- Wireshark adapter for the network anomaly detector HTTP service.

local detector = Proto("ai_anomaly", "AI Network Anomaly")

local score_field = ProtoField.float("ai_anomaly.score", "Anomaly score")
local anomaly_field = ProtoField.bool("ai_anomaly.is_anomaly", "Is anomaly")
local prediction_field = ProtoField.string("ai_anomaly.prediction", "Prediction")
local confidence_field = ProtoField.float("ai_anomaly.confidence", "Confidence")
local flow_field = ProtoField.string("ai_anomaly.flow_id", "Flow ID")

detector.fields = {
    score_field,
    anomaly_field,
    prediction_field,
    confidence_field,
    flow_field,
}

detector.prefs.server_url = Pref.string(
    "Scoring endpoint",
    "http://127.0.0.1:8080/api/v1/captures/score",
    "Local anomaly detector capture-scoring endpoint"
)
detector.prefs.curl_path = Pref.string(
    "curl executable",
    "/usr/bin/curl",
    "Absolute path to curl"
)
detector.prefs.timeout_seconds = Pref.uint(
    "Request timeout in seconds",
    300,
    "Maximum time to wait for capture scoring"
)

local annotations = {}

local function shell_quote(value)
    return "'" .. tostring(value):gsub("'", "'\\''") .. "'"
end

local function split_tsv(line)
    local fields = {}
    local start = 1
    while true do
        local separator = line:find("\t", start, true)
        if separator == nil then
            table.insert(fields, line:sub(start))
            break
        end
        table.insert(fields, line:sub(start, separator - 1))
        start = separator + 1
    end
    return fields
end

local function parse_annotations(body)
    local parsed = {}
    local line_number = 0
    for line in body:gmatch("[^\r\n]+") do
        line_number = line_number + 1
        if line_number > 1 then
            local fields = split_tsv(line)
            if #fields ~= 6 then
                error("Unexpected server response: " .. line)
            end
            local frame_number = tonumber(fields[1])
            local score = tonumber(fields[3])
            if frame_number == nil or score == nil then
                error("Invalid frame number or anomaly score from server")
            end
            parsed[frame_number] = {
                flow_id = fields[2],
                anomaly_score = score,
                is_anomaly = fields[4] == "true",
                prediction = fields[5],
                confidence = tonumber(fields[6]),
            }
        end
    end
    if line_number == 0 then
        error("The scoring server returned an empty response")
    end
    return parsed
end

local function remove_if_present(path)
    if path ~= nil then
        os.remove(path)
    end
end

local function export_capture(capture_path, frames_path)
    local frames, open_error = io.open(frames_path, "w")
    if frames == nil then
        error("Could not create frame manifest: " .. tostring(open_error))
    end

    local dumper = nil
    local packet_count = 0
    local tap = Listener.new()
    function tap.packet(pinfo)
        if dumper == nil then
            dumper = Dumper.new_for_current(capture_path)
        end
        dumper:dump_current()
        frames:write(tostring(pinfo.number), "\n")
        packet_count = packet_count + 1
    end

    local ok, retap_error = pcall(retap_packets)
    tap:remove()
    frames:close()
    if dumper ~= nil then
        dumper:flush()
        dumper:close()
    end
    if not ok then
        error("Could not export the open capture: " .. tostring(retap_error))
    end
    if packet_count == 0 then
        error("The open capture contains no packets")
    end
    return packet_count
end

local function request_scores(capture_path, frames_path)
    if package.config:sub(1, 1) == "\\" then
        error("This plugin currently supports macOS and Linux")
    end
    local server_url = tostring(detector.prefs.server_url)
    if not server_url:match("^https?://") then
        error("The scoring endpoint must start with http:// or https://")
    end

    local command = table.concat({
        shell_quote(detector.prefs.curl_path),
        "--silent",
        "--show-error",
        "--fail-with-body",
        "--max-time",
        tostring(detector.prefs.timeout_seconds),
        "--form",
        shell_quote("capture=@" .. capture_path),
        "--form",
        shell_quote("frames=@" .. frames_path),
        shell_quote(server_url),
        "2>&1",
    }, " ")

    local process, process_error = io.popen(command, "r")
    if process == nil then
        error("Could not start curl: " .. tostring(process_error))
    end
    local body = process:read("*a")
    local ok, reason, status = process:close()
    if ok ~= true then
        error(
            "Scoring request failed (" .. tostring(reason) .. " " ..
            tostring(status) .. "): " .. body
        )
    end
    return parse_annotations(body)
end

local function analyze_open_capture()
    local status = TextWindow.new("AI Network Anomaly Detector")
    status:set_editable(false)
    status:set("Exporting the open capture...\n")

    local base_path = os.tmpname()
    remove_if_present(base_path)
    local capture_path = base_path .. ".pcapng"
    local frames_path = base_path .. ".frames.txt"

    local ok, result = pcall(function()
        local packet_count = export_capture(capture_path, frames_path)
        status:append("Sending " .. tostring(packet_count) .. " packets to the detector...\n")
        return request_scores(capture_path, frames_path)
    end)

    remove_if_present(capture_path)
    remove_if_present(frames_path)

    if not ok then
        status:append("\nAnalysis failed:\n" .. tostring(result) .. "\n")
        return
    end

    annotations = result
    local annotated = 0
    local anomalous = 0
    for _, annotation in pairs(annotations) do
        annotated = annotated + 1
        if annotation.is_anomaly then
            anomalous = anomalous + 1
        end
    end
    status:append(
        "Complete. Annotated " .. tostring(annotated) .. " packets; " ..
        tostring(anomalous) .. " are part of anomalous flows.\n"
    )
    status:append("Use ai_anomaly.is_anomaly == 1 to filter flagged packets.\n")
    redissect_packets()
end

function detector.dissector(_, pinfo, tree)
    local annotation = annotations[tonumber(pinfo.number)]
    if annotation == nil then
        return
    end

    local subtree = tree:add(detector, "AI Network Anomaly")
    subtree:add(score_field, annotation.anomaly_score)
    subtree:add(anomaly_field, annotation.is_anomaly)
    subtree:add(prediction_field, annotation.prediction)
    subtree:add(flow_field, annotation.flow_id)
    if annotation.confidence ~= nil then
        subtree:add(confidence_field, annotation.confidence)
    end
    if annotation.is_anomaly then
        pinfo.cols.info:append(
            string.format(" [AI: %s, score %.4f]", annotation.prediction, annotation.anomaly_score)
        )
    end
end

function detector.init()
    annotations = {}
end

register_postdissector(detector)
if gui_enabled() then
    register_menu(
        "AI Network Anomaly/Analyze Open Capture",
        analyze_open_capture,
        MENU_TOOLS_UNSORTED
    )
end
