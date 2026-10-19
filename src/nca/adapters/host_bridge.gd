extends SceneTree

# This client-side launcher calls accepted public Cybersand owners. It never
# edits a host script, integrates Rapier separately, or reads native world storage.
const PhysicalArena = preload("res://scripts/physical_actor_arena.gd")
const SampledArena = preload("res://scripts/thunderbowl_arena.gd")
const PhysicalCatalog = preload("res://scripts/physical_actor_catalog.gd")
const PROTOCOL: String = "nca-host/1"
const MAX_BYTES: int = 1048576
const MAX_REPEAT: int = 18
const STEP_USEC: float = 1000000.0 / 60.0
const MAX_IDLE_TRACES: int = 36
const OUTPUT_DEADLINE_USEC: int = 2000000
var peer := StreamPeerTCP.new()
var arena: Variant
var lane: String = "act004"
var mode: String = "stepped"
var actor_id: int = 0
var token: String = ""
var pending_bytes := PackedByteArray()
var last_request: int = 0
var configured: bool = false
var next_tick_usec: float = 0.0
var pending_step: Dictionary = {}
var interval_traces: Array = []
var finished: bool = true
var ticks_without_reader: int = 0
var background_traces: Array = []
var background_trace_drops: int = 0
var outbound := PackedByteArray()
var outbound_started_usec: int = 0
var exit_after_flush: int = -1


func _init() -> void:
	var options: Dictionary = {}
	for argument: String in OS.get_cmdline_user_args():
		var separator: int = argument.find("=")
		if not argument.begins_with("--") or separator < 3:
			quit(2)
			return
		options[argument.substr(2, separator - 2)] = argument.substr(separator + 1)
	lane = str(options.get("lane", "act004"))
	mode = str(options.get("mode", "stepped"))
	token = str(options.get("token", ""))
	if lane not in ["act004", "sampled"] or mode not in ["stepped", "realtime"] or token.length() != 64:
		quit(2)
		return
	# The parent owns the loopback listener; no public listening service exists.
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
	arena = PhysicalArena.new() if lane == "act004" else SampledArena.new()
	if arena is Node:
		root.add_child(arena)
	_send({"protocol": PROTOCOL, "id": 0, "ok": true, "token": token,
		"lane": lane, "mode": mode, "godot_version": Engine.get_version_info(),
		"physics_engine": ProjectSettings.get_setting("physics/2d/physics_engine", ""),
		"native_available": ClassDB.class_exists(&"CyberNativeCellWorld"),
		"rapier_available": ClassDB.class_exists(&"RapierPhysicsServer2D"),
		"schedule": PhysicalArena.SCHEDULE if lane == "act004" else "sampled-motor-material/v1",
		"decision_ticks": 6, "helper_lease_ticks": 12, "maximum_lease_ticks": 18,
		"passive_wrench_lanes": ["sampled-metal"]})


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
	# One parsed request per iteration, with a bounded input queue. Already-read
	# packets are handled even when the next TCP poll returns zero new bytes.
	if newline >= 0:
		var bytes: PackedByteArray = pending_bytes.slice(0, newline)
		pending_bytes = pending_bytes.slice(newline + 1)
		var parser := JSON.new()
		if parser.parse(bytes.get_string_from_utf8()) != OK or not parser.data is Dictionary:
			_fatal("packet_json")
			return false
		_handle(parser.data)
	if mode == "realtime" and configured and not finished:
		var now: float = float(Time.get_ticks_usec())
		var count: int = 0
		while now >= next_tick_usec and not finished and count < 8:
			var commands: Array = pending_step.get("commands", [])
			pending_step.erase("commands")
			var trace: Dictionary = _tick(commands)
			if not pending_step.is_empty():
				interval_traces.append(trace)
				ticks_without_reader = 0
			else:
				ticks_without_reader += 1
				if background_traces.size() >= MAX_IDLE_TRACES:
					background_traces.pop_front()
					background_trace_drops += 1
				background_traces.append(trace)
			next_tick_usec += STEP_USEC
			count += 1
			if not pending_step.is_empty() and (finished or interval_traces.size() >= int(pending_step.ticks)):
				var request: Dictionary = pending_step
				pending_step = {}
				_reply_transition(int(request.id), interval_traces, int(request.start_tick))
				interval_traces = []
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
		"reset": ["protocol", "id", "op", "config", "actor_id"],
		"observe": ["protocol", "id", "op"], "profile": ["protocol", "id", "op"],
		"step": ["protocol", "id", "op", "commands", "ticks"],
		"cancel": ["protocol", "id", "op", "run_id", "epoch"],
		"close": ["protocol", "id", "op"]}
	if not request_keys.has(operation):
		_error(request_id, "unknown_operation")
		return
	for key: Variant in request:
		if key not in request_keys[operation]:
			_error(request_id, "unknown_request_field")
			return
	if not pending_step.is_empty():
		_fatal("one_request_in_flight")
		return
	match operation:
		"reset":
			var config: Variant = request.get("config")
			var assigned: Variant = request.get("actor_id", 0)
			if not config is Dictionary or not _integer(assigned, 0, 7) \
				or not config.get("actors") is Array or int(assigned) >= config.actors.size():
				_error(request_id, "reset_schema")
				return
			if not config.actors[int(assigned)] is Dictionary:
				_error(request_id, "reset_actor_schema")
				return
			var definition: Dictionary = config.actors[int(assigned)]
			if lane == "act004" and str(definition.get("producer", "")) != "policy":
				_error(request_id, "assigned_actor_requires_policy")
				return
			if lane == "sampled" and str(definition.get("controller", "")).begins_with("human_"):
				_error(request_id, "assigned_actor_requires_policy")
				return
			if not arena.configure(config):
				_error(request_id, "configure_refused: " + str(arena.last_error))
				return
			actor_id = int(assigned)
			configured = true
			finished = false
			next_tick_usec = float(Time.get_ticks_usec()) + STEP_USEC
			ticks_without_reader = 0
			background_traces = []
			background_trace_drops = 0
			_reply_transition(request_id, [], 0)
		"observe":
			if not configured:
				_error(request_id, "not_configured")
				return
			_reply_transition(request_id, [], int(arena.snapshot().tick), false)
		"profile":
			if lane != "act004" or not configured:
				_error(request_id, "physical_profile_unavailable")
				return
			var view: Dictionary = arena.observation(actor_id)
			_reply(request_id, {"profile": PhysicalCatalog.entry(StringName(str(view.model)))})
		"step":
			if not configured:
				_error(request_id, "episode_finished_or_not_configured")
				return
			var ticks: Variant = request.get("ticks", 6)
			var commands: Variant = request.get("commands", [])
			if not _integer(ticks, 1, MAX_REPEAT) or not commands is Array or commands.size() > 1:
				_error(request_id, "step_schema")
				return
			for command: Variant in commands:
				# Address fencing precedes owner admission; the client controls only its assignment.
				if not command is Dictionary or command.get("actor_id") != actor_id:
					_error(request_id, "actor_assignment")
					return
			var start_tick: int = int(arena.snapshot().tick)
			if finished:
				if mode == "realtime":
					# The host may reach its horizon while inference is running.
					# Return that final state once through the ordinary step path;
					# this late request was never consumed by a game tick.
					_reply_transition(request_id, [], start_tick, true, commands)
				else:
					_error(request_id, "episode_finished_or_not_configured")
				return
			if mode == "realtime":
				pending_step = {"id": request_id, "ticks": int(ticks), "commands": commands, "start_tick": start_tick}
				interval_traces = []
			else:
				var traces: Array = []
				for index: int in range(int(ticks)):
					traces.append(_tick(commands if index == 0 else []))
					if finished:
						break
				_reply_transition(request_id, traces, start_tick)
		"cancel":
			if not configured or lane != "act004":
				_error(request_id, "physical_cancel_unavailable")
				return
			if request.get("run_id") != arena.session.run_id or request.get("epoch") != arena.session.epoch:
				_error(request_id, "cancel_identity")
				return
			var accepted: bool = arena.cancel(actor_id, &"policy")
			_reply(request_id, {"cancelled": accepted, "observation": arena.observation(actor_id)})
		"close":
			_close()
			_reply(request_id, {"closed": true})
			exit_after_flush = 0
		_:
			_error(request_id, "unknown_operation")


func _tick(commands: Array) -> Dictionary:
	var before: Dictionary = arena.snapshot()
	var start_usec: int = Time.get_ticks_usec()
	var state: Dictionary = arena.step(commands)
	finished = bool(state.get("finished", false))
	var view: Dictionary = arena.observation(actor_id)
	var trace: Dictionary = {"tick_before": int(before.tick), "tick_after": int(state.tick),
		"world_tick": state.get("world_tick", state.tick), "received_monotonic_usec": start_usec,
		"completed_monotonic_usec": Time.get_ticks_usec(), "requested": commands.duplicate(true),
		"admission": state.get("command_results", state.get("admission", {})),
		"observation": view, "failed": bool(state.get("failed", false))}
	if lane == "act004":
		trace["applied"] = {"action": null if finished else view.get("action", {}),
			"action_available": not finished, "terminal_revocation": finished,
			"linear_impulse": view.get("applied_impulse"),
			"angular_impulse": view.get("applied_torque_impulse"), "energy_used": view.get("energy_used")}
		trace["passive"] = {"available": false,
			"reason": "ACT004 publishes aggregate actuation and actual motion, not a separate passive wrench"}
		trace["actual_motion"] = {"position": view.get("position"), "velocity": view.get("velocity"),
			"rotation": view.get("rotation"), "angular_velocity": view.get("angular_velocity")}
		trace["cellular_sample_tick"] = state.get("cellular_sample_tick")
	else:
		var packed: Array = view.get("actuation_state", [])
		var metal: bool = str(view.get("control_mode", "")) == "metal" and packed.size() == 32
		trace["applied"] = {"actuation_state": packed, "available": metal}
		var context: Dictionary = state.get("actor_prototypes", {})
		var consumed: PackedFloat32Array = context.get("applied_actions", PackedFloat32Array())
		if not metal and not finished and consumed.size() >= (actor_id + 1) * 10:
			trace["applied"] = {"available": true, "wire_values": consumed.slice(actor_id * 10, (actor_id + 1) * 10),
				"sequence": context.applied_sequences[actor_id], "flags": context.applied_flags[actor_id],
				"source": "actor session consumed wire values: move2,aim2,effort,trim,reel,primary,secondary,auxiliary"}
		if metal:
			trace["passive"] = {"available": true, "force": [packed[14], packed[15]], "torque": packed[16],
				"source": "ACT003 actuation_state slots14..16; sampled, not Rapier"}
		else:
			trace["passive"] = {"available": false, "reason": "ACT003 wire has no direct-actuation wrench receipt"}
	return trace


func _reply_transition(request_id: int, traces: Array, start_tick: int, consume_background: bool = true,
		unconsumed_commands: Array = []) -> void:
	var state: Dictionary = arena.snapshot()
	var failed: bool = bool(state.get("failed", false))
	var reason: String = str(state.get("reason", ""))
	var cumulative: Variant = null
	if lane == "sampled":
		cumulative = state.rewards[actor_id]
	_reply(request_id, {"observation": arena.observation(actor_id), "traces": traces,
		"start_tick": start_tick, "tick": int(state.tick), "actual_ticks": int(state.tick) - start_tick,
		"finished": bool(state.get("finished", false)), "failed": failed, "reason": reason,
		"terminated": bool(state.get("terminated", bool(state.get("finished", false)) and reason != "tick_limit")),
		"truncated": bool(state.get("truncated", reason == "tick_limit")),
		"cumulative_reward": cumulative, "reward_kind": "cumulative" if lane == "sampled" else "no_task_reward",
		"schedule": PhysicalArena.SCHEDULE if lane == "act004" else "sampled-motor-material/v1",
		"world_tick": state.get("world_tick", state.tick), "ticks_without_reader": ticks_without_reader,
		"idle_trace": background_traces.duplicate(true), "idle_trace_dropped": background_trace_drops,
		"unconsumed_terminal_commands": unconsumed_commands})
	if consume_background:
		background_traces = []
		background_trace_drops = 0


func _reply(request_id: int, body: Dictionary) -> void:
	body.merge({"protocol": PROTOCOL, "id": request_id, "ok": true}, true)
	_send(body)


func _error(request_id: int, reason: String) -> void:
	_send({"protocol": PROTOCOL, "id": request_id, "ok": false, "error": reason})


func _fatal(reason: String) -> void:
	_error(last_request, reason)
	_close()
	exit_after_flush = 2


func _send(value: Dictionary) -> void:
	var data: PackedByteArray = (JSON.stringify(PhysicalArena.json_value(value)) + "\n").to_utf8_buffer()
	if data.size() > MAX_BYTES:
		_close()
		quit(2)
		return
	if not outbound.is_empty():
		# A producer cannot accumulate responses behind a stalled reader.
		_close()
		quit(2)
		return
	outbound = data
	outbound_started_usec = Time.get_ticks_usec()
	_flush_output()


func _flush_output() -> void:
	if not outbound.is_empty():
		if Time.get_ticks_usec() - outbound_started_usec >= OUTPUT_DEADLINE_USEC:
			_close()
			exit_after_flush = 2
			quit(2)
			return
		# put_data is blocking. Partial writes preserve realtime advancement even
		# when the receiving process has stopped draining its socket.
		var sent: Array = peer.put_partial_data(outbound)
		if int(sent[0]) != OK:
			_close()
			exit_after_flush = 2
			quit(2)
			return
		outbound = outbound.slice(int(sent[1]))
	if outbound.is_empty() and exit_after_flush >= 0:
		quit(exit_after_flush)


static func _integer(value: Variant, minimum: int, maximum: int) -> bool:
	return (value is int or value is float) and is_finite(float(value)) \
		and float(value) == floorf(float(value)) and float(value) >= minimum and float(value) <= maximum


func _close() -> void:
	configured = false
	finished = true
	if arena != null:
		arena.close()
		if arena is Node:
			arena.free()
		arena = null
