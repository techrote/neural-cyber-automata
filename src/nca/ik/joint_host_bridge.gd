extends SceneTree

# NCA-owned external bridge for the separately negotiated Cybersand
# physical_joint_target/v1 lane.  It calls only public session methods and
# never edits host source, assigns body transforms, or integrates physics itself.
const Session = preload("res://scripts/physical_actor_session.gd")
const PROTOCOL: String = "nca-host/1"
const MAX_BYTES: int = 1048576
const MAX_TICKS: int = 18
const DT: float = 1.0 / 60.0
const OUTPUT_DEADLINE_USEC: int = 2000000

var peer := StreamPeerTCP.new()
var pending_bytes := PackedByteArray()
var outbound := PackedByteArray()
var outbound_started_usec: int = 0
var exit_after_flush: int = -1
var last_request: int = 0
var token: String = ""
var source_commit: String = ""
var runtime_sha256: String = ""
var owner: Node2D
var session: CyberPhysicalActorSession
var fixture_bodies: Array[RID] = []
var fixture_shapes: Array[RID] = []


func _init() -> void:
	var options: Dictionary = {}
	for argument: String in OS.get_cmdline_user_args():
		var separator: int = argument.find("=")
		if not argument.begins_with("--") or separator < 3:
			quit(2)
			return
		options[argument.substr(2, separator - 2)] = argument.substr(separator + 1)
	token = str(options.get("token", ""))
	source_commit = str(options.get("source-commit", ""))
	runtime_sha256 = str(options.get("runtime-sha256", ""))
	if token.length() != 64 or not _hex(source_commit, 40) or not _hex(runtime_sha256, 64):
		quit(2)
		return
	if peer.connect_to_host("127.0.0.1", int(options.get("port", 0))) != OK:
		quit(2)
		return
	call_deferred("_start")


func _start() -> void:
	peer.poll()
	if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		quit(2)
		return
	peer.set_no_delay(true)
	if not ClassDB.class_exists(&"RapierPhysicsServer2D"):
		_fatal("rapier_unavailable")
		return
	owner = Node2D.new()
	root.add_child(owner)
	_make_contact_fixture(owner.get_world_2d().space)
	session = Session.new()
	owner.add_child(session)
	var definition: Dictionary = {
		"model": &"runner", "producer": &"policy", "position": Vector2(400, 400),
		"ik_pilot": true, "usable_index": -1
	}
	if not session.configure([definition], owner.get_world_2d().space, 1,
			{"source_commit": source_commit, "runtime_sha256": runtime_sha256}):
		_fatal("ik_session_configure_refused:" + str(session.last_error))
		return
	var capability: Dictionary = session.ik_capabilities(0)
	if capability.is_empty():
		_fatal("ik_capability_unavailable")
		return
	_send({"protocol": PROTOCOL, "id": 0, "ok": true, "token": token,
		"lane": "physical_joint_target/v1",
		"host_source_commit": source_commit, "host_runtime_sha256": runtime_sha256,
		"rapier_available": true, "godot_version": Engine.get_version_info()})


func _make_contact_fixture(space: RID) -> void:
	# A fixed floor and side stop exercise real support/contact and blocked motion
	# in the adapter conformance suite.  They are test-world geometry, not a
	# public perturbation or force API.
	var floor_shape: RID = PhysicsServer2D.rectangle_shape_create()
	PhysicsServer2D.shape_set_data(floor_shape, Vector2(80, 1))
	var wall_shape: RID = PhysicsServer2D.rectangle_shape_create()
	PhysicsServer2D.shape_set_data(wall_shape, Vector2(1, 18))
	var body: RID = PhysicsServer2D.body_create()
	PhysicsServer2D.body_set_mode(body, PhysicsServer2D.BODY_MODE_STATIC)
	PhysicsServer2D.body_add_shape(body, floor_shape, Transform2D(0, Vector2(400, 411)))
	PhysicsServer2D.body_add_shape(body, wall_shape, Transform2D(0, Vector2(408, 394)))
	PhysicsServer2D.body_set_space(body, space)
	fixture_shapes = [floor_shape, wall_shape]
	fixture_bodies = [body]


func _process(_delta: float) -> bool:
	peer.poll()
	if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		_close()
		quit(0)
		return false
	_flush_output()
	if exit_after_flush >= 0:
		return false
	var available: int = peer.get_available_bytes()
	if available > 0:
		if pending_bytes.size() + available > MAX_BYTES:
			_fatal("packet_capacity")
			return false
		var packet: Array = peer.get_data(available)
		if int(packet[0]) != OK:
			_fatal("packet_read")
			return false
		pending_bytes.append_array(packet[1])
	var newline: int = pending_bytes.find(10)
	if newline >= 0:
		var bytes: PackedByteArray = pending_bytes.slice(0, newline)
		pending_bytes = pending_bytes.slice(newline + 1)
		var parser := JSON.new()
		if parser.parse(bytes.get_string_from_utf8()) != OK or not parser.data is Dictionary:
			_fatal("packet_json")
			return false
		_handle(parser.data)
	_flush_output()
	return false


func _handle(request: Dictionary) -> void:
	if request.get("protocol") != PROTOCOL or not _integer(request.get("id"), 1, 9007199254740991):
		_fatal("protocol_identity")
		return
	var request_id: int = int(request.id)
	if request_id <= last_request:
		_fatal("request_order")
		return
	last_request = request_id
	var operation: String = str(request.get("op", ""))
	var request_keys: Dictionary = {
		"ik_capabilities": ["protocol", "id", "op"],
		"ik_grant": ["protocol", "id", "op", "source_id", "policy_sha256", "goal_id", "goal_deadline_tick"],
		"ik_observe": ["protocol", "id", "op", "grant_id"],
		"ik_submit": ["protocol", "id", "op", "packet"],
		"ik_receipts": ["protocol", "id", "op", "grant_id", "sequence"],
		"advance": ["protocol", "id", "op", "ticks"],
		"ik_cancel": ["protocol", "id", "op", "grant_id", "source_id"],
		"close": ["protocol", "id", "op"]
	}
	if not request_keys.has(operation):
		_error(request_id, "unknown_operation")
		return
	for key: Variant in request:
		if key not in request_keys[operation]:
			_error(request_id, "unknown_request_field")
			return
	match operation:
		"ik_capabilities":
			var capability: Dictionary = session.ik_capabilities(0)
			if capability.is_empty():
				_error(request_id, "ik_capability_unavailable")
				return
			_reply(request_id, {"capability": capability})
		"ik_grant":
			var grant: Dictionary = session.ik_grant(0, &"policy",
				str(request.get("source_id", "")), str(request.get("policy_sha256", "")),
				str(request.get("goal_id", "")), int(request.get("goal_deadline_tick", -1)))
			if grant.is_empty():
				_error(request_id, "ik_grant_refused:" + str(session.last_error))
				return
			_reply(request_id, {"grant": grant})
		"ik_observe":
			var observation: Dictionary = session.ik_observe(0, str(request.get("grant_id", "")))
			if observation.is_empty():
				_error(request_id, "ik_observation_refused")
				return
			_reply(request_id, {"observation": observation})
		"ik_submit":
			var packet: Variant = _canonical_joint_packet(request.get("packet"))
			var admission: Dictionary = session.ik_submit(packet)
			# Refusals are protocol results, not transport failures.
			_reply(request_id, {"admission": admission})
		"ik_receipts":
			var sequence: Variant = request.get("sequence")
			if not _integer(sequence, -1, 9007199254740991):
				_error(request_id, "ik_receipt_sequence")
				return
			_reply(request_id, {"receipts": session.ik_receipts(
				0, str(request.get("grant_id", "")), int(sequence))})
		"advance":
			var ticks: Variant = request.get("ticks")
			if not _integer(ticks, 1, MAX_TICKS):
				_error(request_id, "advance_ticks")
				return
			for _index: int in range(int(ticks)):
				session.advance(DT)
				session.bridge.step()
				session.complete_step()
				if not session.active():
					_error(request_id, "host_session_fault:" + str(session.last_error))
					return
			_reply(request_id, {"ticks": int(ticks), "observation": session.observation(0)})
		"ik_cancel":
			var cancellation: Dictionary = session.ik_cancel(
				0, str(request.get("grant_id", "")), str(request.get("source_id", "")))
			_reply(request_id, {"cancellation": cancellation})
		"close":
			_close()
			_reply(request_id, {"closed": true})
			exit_after_flush = 0
		_:
			_error(request_id, "unknown_operation")


func _reply(request_id: int, payload: Dictionary) -> void:
	var result: Dictionary = {"protocol": PROTOCOL, "id": request_id, "ok": true}
	result.merge(payload)
	_send(result)


func _error(request_id: int, reason: String) -> void:
	_send({"protocol": PROTOCOL, "id": request_id, "ok": false, "error": reason})


func _fatal(reason: String) -> void:
	if last_request > 0:
		_error(last_request, reason)
	else:
		_send({"protocol": PROTOCOL, "id": 0, "ok": false, "error": reason, "token": token})
	exit_after_flush = 2


func _send(value: Dictionary) -> void:
	var data: PackedByteArray = (JSON.stringify(_json_value(value)) + "\n").to_utf8_buffer()
	if data.size() > MAX_BYTES:
		_fatal("response_capacity")
		return
	outbound.append_array(data)
	if outbound_started_usec == 0:
		outbound_started_usec = Time.get_ticks_usec()


func _flush_output() -> void:
	if outbound.is_empty():
		if exit_after_flush >= 0:
			var code: int = exit_after_flush
			exit_after_flush = -1
			quit(code)
		return
	peer.poll()
	if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		_close()
		quit(2)
		return
	var result: Array = peer.put_partial_data(outbound)
	if int(result[0]) != OK:
		_close()
		quit(2)
		return
	var written: int = int(result[1])
	if written > 0:
		outbound = outbound.slice(written)
		if outbound.is_empty():
			outbound_started_usec = 0
	if not outbound.is_empty() and Time.get_ticks_usec() - outbound_started_usec > OUTPUT_DEADLINE_USEC:
		_close()
		quit(2)


func _json_value(value: Variant) -> Variant:
	if value is Vector2:
		return [float(value.x), float(value.y)]
	if value is StringName:
		return str(value)
	if value is RID:
		return null
	if value is Dictionary:
		var result: Dictionary = {}
		for key: Variant in value:
			result[str(key)] = _json_value(value[key])
		return result
	if value is Array:
		var result: Array = []
		for item: Variant in value:
			result.append(_json_value(item))
		return result
	return value


func _canonical_joint_packet(value: Variant) -> Variant:
	# JSON has one numeric type at the wire boundary.  Canonicalize only fields
	# whose public Cybersand schema is integral; preserve every other field and
	# every malformed value so the owner remains the authority that refuses it.
	if not value is Dictionary:
		return value
	var packet: Dictionary = value.duplicate(true)
	for key: String in [
		"version", "actor_id", "epoch", "incarnation", "control_generation",
		"grant_generation", "goal_deadline_tick", "sequence", "observed_tick",
		"lease_ticks"
	]:
		if packet.has(key) and _integer(packet[key], -9007199254740991, 9007199254740991):
			packet[key] = int(packet[key])
	return packet


func _integer(value: Variant, minimum: int, maximum: int) -> bool:
	return (value is int or value is float) and is_finite(float(value)) \
		and float(value) == floorf(float(value)) and float(value) >= minimum and float(value) <= maximum


func _hex(value: String, length: int) -> bool:
	if value.length() != length:
		return false
	for character: String in value:
		if character not in "0123456789abcdef":
			return false
	return true


func _close() -> void:
	if is_instance_valid(session):
		session.close()
		session.free()
		session = null
	if is_instance_valid(owner):
		owner.free()
	for body: RID in fixture_bodies:
		if body.is_valid():
			PhysicsServer2D.free_rid(body)
	for shape: RID in fixture_shapes:
		if shape.is_valid():
			PhysicsServer2D.free_rid(shape)
	fixture_bodies.clear()
	fixture_shapes.clear()
