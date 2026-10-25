extends SceneTree

# 4LY-004 external transport around the real Cybersand flight episode owner.
# This file lives in neural-cyber-automata and is passed to Godot with --script.
# It does not patch Cybersand, reproduce physics, or create a second simulator.
const FlightEpisode = preload("res://scripts/physical_flight_episode.gd")
const FlightProfile = preload("res://scripts/physical_flight_profile.gd")
const FlightController = preload("res://scripts/physical_flight_controller.gd")
const FlightCourse = preload("res://scripts/physical_flight_course.gd")
const FlightObservation = preload("res://scripts/physical_flight_observation_adapter.gd")
const PhysicalSession = preload("res://scripts/physical_actor_session.gd")
const PhysicalArena = preload("res://scripts/physical_actor_arena.gd")

const PROTOCOL: String = "nca-4ly-flight/1"
const MAX_BYTES: int = 1048576
const OUTPUT_DEADLINE_USEC: int = 2000000
const TRAINER: StringName = &"4ly_trainer"
const ACTOR_ID: int = 0

var peer := StreamPeerTCP.new()
var token: String = ""
var pending_bytes := PackedByteArray()
var outbound := PackedByteArray()
var outbound_started_usec: int = 0
var exit_after_flush: int = -1
var last_request: int = 0
var episode: CyberPhysicalFlightEpisode


func _init() -> void:
	var options: Dictionary = {}
	for argument: String in OS.get_cmdline_user_args():
		var separator: int = argument.find("=")
		if not argument.begins_with("--") or separator < 3:
			quit(2)
			return
		options[argument.substr(2, separator - 2)] = argument.substr(separator + 1)
	token = str(options.get("token", ""))
	if token.length() != 64:
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
	episode = FlightEpisode.new()
	root.add_child(episode)
	var profile: Dictionary = FlightProfile.entry(TRAINER)
	_send({
		"protocol": PROTOCOL, "id": 0, "ok": true, "token": token,
		"episode_version": FlightEpisode.VERSION,
		"observation_version": FlightEpisode.OBS_VERSION,
		"flat_observation_version": FlightObservation.VERSION,
		"flat_observation_length": FlightObservation.LENGTH,
		"action_version": FlightEpisode.ACTION_VERSION,
		"action_profile": FlightProfile.ACTION_PROFILE,
		"course_id": FlightCourse.ID,
		"course_version": FlightCourse.VERSION,
		"split_geometry_version": FlightCourse.SPLIT_VERSION,
		"host_schedule": PhysicalArena.SCHEDULE,
		"physics_hz": PhysicalArena.PHYSICS_HZ,
		"decision_ticks": FlightEpisode.DECISION_TICKS,
		"lease_ticks": FlightEpisode.LEASE_TICKS,
		"actor_id": ACTOR_ID,
		"producer": "policy",
		"teacher_capture": "owner_reference_action_development_only",
		"trainer_profile_sha256": FlightProfile.profile_hash(profile),
		"controller_sha256": FlightController.controller_hash(),
		"native_coupling": true,
	})


func _process(_delta: float) -> bool:
	peer.poll()
	if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		_close_owner()
		quit(0)
		return false
	_flush_output()
	if exit_after_flush >= 0:
		return false

	var available: int = peer.get_available_bytes()
	if available > 0:
		if pending_bytes.size() + available > MAX_BYTES:
			_fatal("packet_capacity", "none")
			return false
		var packet: Array = peer.get_data(available)
		if int(packet[0]) != OK:
			_fatal("packet_read", "none")
			return false
		pending_bytes.append_array(packet[1])

	var newline: int = pending_bytes.find(10)
	if newline >= 0:
		var bytes: PackedByteArray = pending_bytes.slice(0, newline)
		pending_bytes = pending_bytes.slice(newline + 1)
		var parser := JSON.new()
		if parser.parse(bytes.get_string_from_utf8()) != OK or not parser.data is Dictionary:
			_fatal("packet_json", "none")
			return false
		_handle(parser.data)
	_flush_output()
	return false


func _handle(request: Dictionary) -> void:
	if request.get("protocol") != PROTOCOL or not _integer(request.get("id"), 1, 9007199254740991):
		_fatal("protocol_identity", "none")
		return
	var request_id: int = int(request.id)
	if request_id <= last_request:
		# A replayed ID could refer to already-consumed work. Do not pretend it
		# is safely retryable.
		_error(request_id, "request_order_or_replay", "unknown")
		_close_owner()
		exit_after_flush = 2
		return
	last_request = request_id
	var operation: String = str(request.get("op", ""))
	var request_keys: Dictionary = {
		"reset": ["protocol", "id", "op", "profile", "producer", "scenario", "seed", "split"],
		"step": ["protocol", "id", "op", "run_id", "actor_id", "epoch", "control_generation",
			"observation_id", "physical_tick", "action", "repeat_ticks"],
		"teacher_step": ["protocol", "id", "op", "run_id", "actor_id", "epoch", "control_generation",
			"observation_id", "physical_tick", "repeat_ticks"],
		"cancel": ["protocol", "id", "op", "run_id", "actor_id", "epoch", "control_generation",
			"observation_id", "physical_tick", "producer"],
		"takeover": ["protocol", "id", "op", "run_id", "actor_id", "epoch", "control_generation",
			"observation_id", "physical_tick", "producer"],
		"close": ["protocol", "id", "op"],
	}
	if not request_keys.has(operation):
		_error(request_id, "unknown_operation", "none")
		return
	for key: Variant in request:
		if key not in request_keys[operation]:
			_error(request_id, "unknown_request_field", "none")
			return
	for required: String in request_keys[operation]:
		if not request.has(required):
			_error(request_id, "missing_request_field", "none")
			return

	match operation:
		"reset":
			_reset(request_id, request)
		"step":
			_step(request_id, request)
		"teacher_step":
			_teacher_step(request_id, request)
		"cancel":
			_revoke(request_id, request, false)
		"takeover":
			_revoke(request_id, request, true)
		"close":
			_close_owner()
			_reply(request_id, {"closed": true}, "confirmed")
			exit_after_flush = 0
		_:
			_error(request_id, "unknown_operation", "none")


func _reset(request_id: int, request: Dictionary) -> void:
	if request.profile != str(TRAINER) or request.producer not in ["policy", "teacher"]:
		_error(request_id, "trainer_source_scope_required", "none")
		return
	if not request.scenario is String or str(request.scenario).is_empty() \
		or not _integer(request.seed, 0, 2147483647) \
		or request.split not in ["development", "tuning"]:
		_error(request_id, "reset_schema_or_excluded_split", "none")
		return
	if request.producer == "teacher" and request.split != "development":
		_error(request_id, "teacher_capture_development_only", "none")
		return
	# Resolve only public development/tuning metadata before touching the owner.
	# Final is excluded above, so this never opens the sealed suite.
	if FlightCourse.scenario(
		str(request.scenario), int(request.seed), str(request.split)).is_empty():
		_error(request_id, "unknown_flight_scenario", "none")
		return
	if not episode.reset(
		TRAINER, str(request.scenario), int(request.seed), str(request.split),
		StringName(str(request.producer))):
		# reset() can fail after arena.configure() (for example during a later
		# owner/observation step). Once called, we cannot prove the replacement
		# owner consumed nothing, so automatic replay is forbidden.
		_error(request_id, "episode_reset_refused: " + episode.last_error, "unknown")
		_close_owner()
		exit_after_flush = 2
		return
	var transition: Dictionary = episode.transition(0)
	_reply_transition(request_id, transition, true)


func _step(request_id: int, request: Dictionary) -> void:
	var current: Dictionary = episode.observe()
	if current.is_empty():
		_error(request_id, "episode_not_live", "none")
		return
	if not _matches_authority(request, current):
		_error(request_id, "stale_or_foreign_authority", "none")
		return
	if episode.source != &"policy":
		_error(request_id, "policy_owner_required", "none")
		return
	if not _integer(request.repeat_ticks, 1, FlightEpisode.DECISION_TICKS):
		_error(request_id, "decision_cadence", "none")
		return
	var repeat_ticks: int = int(request.repeat_ticks)
	var offset: int = int(current.observed_tick) % FlightEpisode.DECISION_TICKS
	if offset + repeat_ticks > FlightEpisode.DECISION_TICKS:
		_error(request_id, "decision_cadence", "none")
		return
	var action: Variant = request.get("action")
	if action == null:
		if not bool(current.get("command_live", false)) \
				or int(current.get("lease_remaining_ticks", 0)) <= 0:
			_error(request_id, "live_lease_required", "none")
			return
	else:
		if offset != 0:
			_error(request_id, "action_outside_decision_boundary", "none")
			return
		if not _valid_action(action):
			_error(request_id, "action_schema", "none")
			return
	var transition: Dictionary = episode.tick(
		action, str(request.observation_id) if action != null else "", repeat_ticks)
	# The owner reports exact physical consumption, including partial terminal
	# intervals and zero-tick authoritative failures.
	_reply_transition(request_id, transition, false)


func _teacher_step(request_id: int, request: Dictionary) -> void:
	var current: Dictionary = episode.observe()
	if current.is_empty():
		_error(request_id, "episode_not_live", "none")
		return
	if not _matches_authority(request, current):
		_error(request_id, "stale_or_foreign_authority", "none")
		return
	if episode.source != &"teacher":
		_error(request_id, "teacher_owner_required", "none")
		return
	if not _integer(request.repeat_ticks, FlightEpisode.DECISION_TICKS, FlightEpisode.DECISION_TICKS) \
		or int(current.observed_tick) % FlightEpisode.DECISION_TICKS != 0:
		_error(request_id, "decision_cadence", "none")
		return

	# The teacher action is computed by the real #229 episode owner from the
	# same copied policy observation. No Python or bridge-side navigation logic
	# exists here.
	var requested: Dictionary = episode.reference_action()
	var admitted: Dictionary = PhysicalSession.normalized_flight_action(requested)
	if requested.is_empty() or admitted.is_empty():
		_error(request_id, "owner_reference_action_invalid", "unknown")
		_close_owner()
		exit_after_flush = 2
		return
	var transition: Dictionary = episode.tick(
		admitted, str(request.observation_id), int(request.repeat_ticks))
	var manifest: Dictionary = episode.manifest()
	var teacher_sha: Variant = manifest.get("teacher_source_sha256")
	if not teacher_sha is String or str(teacher_sha).length() != 64:
		_error(request_id, "teacher_source_identity_missing", "unknown")
		_close_owner()
		exit_after_flush = 2
		return
	var accepted: bool = int(transition.get("ticks_consumed", 0)) > 0
	if not accepted:
		# A zero-tick owner refusal is retained as a failed capture interval, but
		# it is never relabelled as an admitted supervised action.
		_reply_transition(request_id, transition, false, {
			"teacher_action": {
				"source_sha256": str(teacher_sha),
				"observation_id": str(current.observation_id),
				"status": "refused",
				"requested_action": PhysicalArena.json_value(requested.move),
				"admitted_action": null,
				"reason": str(transition.get("reason", "owner_action_refused")),
			}
		})
		return
	_reply_transition(request_id, transition, false, {
		"teacher_action": {
			"source_sha256": str(teacher_sha),
			"observation_id": str(current.observation_id),
			"status": "accepted",
			"requested_action": PhysicalArena.json_value(requested.move),
			"admitted_action": PhysicalArena.json_value(admitted.move),
			"reason": "owner_reference_action_admitted",
		}
	})


func _revoke(request_id: int, request: Dictionary, takeover: bool) -> void:
	var current: Dictionary = episode.observe()
	if current.is_empty():
		_error(request_id, "episode_not_live", "none")
		return
	if not _matches_authority(request, current):
		_error(request_id, "stale_or_foreign_authority", "none")
		return
	var producer: String = str(request.producer)
	if (takeover and producer != "human") or (not takeover and producer != "policy"):
		_error(request_id, "unsupported_revocation_producer", "none")
		return
	var accepted: bool
	if takeover:
		accepted = episode.arena != null and episode.arena.session != null \
			and episode.arena.session.takeover(ACTOR_ID, int(current.epoch), &"human")
	else:
		accepted = episode.arena != null and episode.arena.cancel(ACTOR_ID, &"policy")
	if not accepted:
		_error(request_id, "owner_revocation_refused", "none")
		return
	var owner: Dictionary = episode.arena.observation(ACTOR_ID)
	if owner.is_empty():
		_error(request_id, "owner_revocation_receipt_missing", "unknown")
		_close_owner()
		exit_after_flush = 2
		return
	_reply(request_id, {
		"authority": {
			"version": FlightEpisode.OBS_VERSION,
			"action_version": FlightEpisode.ACTION_VERSION,
			"run_id": str(owner.run_id),
			"actor_id": int(owner.actor_id),
			"epoch": int(owner.epoch),
			"observed_tick": int(owner.tick),
			"observation_id": str(owner.observation_id),
			"control_generation": int(owner.control_generation),
			"producer": str(owner.producer),
		}
	}, "confirmed")


func _matches_authority(request: Dictionary, current: Dictionary) -> bool:
	return request.get("run_id") == current.get("run_id") \
		and request.get("actor_id") == current.get("actor_id") \
		and request.get("epoch") == current.get("epoch") \
		and request.get("control_generation") == current.get("control_generation") \
		and request.get("observation_id") == current.get("observation_id") \
		and request.get("physical_tick") == current.get("observed_tick")


func _reply_transition(request_id: int, transition: Dictionary, include_manifest: bool,
		extra: Dictionary = {}) -> void:
	var observation: Dictionary = transition.get("observation", {})
	var flat: Dictionary = episode.flat_observation()
	if not flat.is_empty() and not observation.is_empty():
		# Transport correlation metadata; not a new Cybersand observation field.
		flat["source_observation_id"] = str(observation.get("observation_id", ""))
	var record: Dictionary = episode.record()
	var trace: Array = record.get("trace", [])
	var consumed: int = int(transition.get("ticks_consumed", 0))
	var receipts: Array = []
	if consumed > 0 and trace.size() >= consumed:
		for index: int in range(trace.size() - consumed, trace.size()):
			receipts.append(trace[index].duplicate(true))
	var body: Dictionary = {
		"transition": transition.duplicate(true),
		"flat_observation": flat.duplicate(true),
		"owner_receipts": receipts,
		"owner_timing": {
			"reset_microseconds": int(record.get("reset_microseconds", 0)),
			"observation_serialization_microseconds":
				int(record.get("observation_serialization_microseconds", 0)),
			"process_wall_microseconds":
				int(record.get("process_wall_microseconds", 0)),
		},
		"owner_metrics": record.get("metrics", {}).duplicate(true),
	}
	for key: Variant in extra:
		if body.has(key):
			_error(request_id, "duplicate_reply_field", "unknown")
			_close_owner()
			exit_after_flush = 2
			return
		body[key] = extra[key]
	if include_manifest:
		body["manifest"] = episode.manifest()
	_reply(request_id, body, "confirmed")


static func _valid_action(value: Variant) -> bool:
	if not value is Dictionary or value.size() != 1 or not value.has("move"):
		return false
	var move: Variant = value.move
	if not move is Array or move.size() != 2:
		return false
	for component: Variant in move:
		if not (component is int or component is float) or not is_finite(float(component)) \
			or absf(float(component)) > 1.0:
			return false
	return true


func _reply(request_id: int, body: Dictionary, consumption: String) -> void:
	body.merge({
		"protocol": PROTOCOL, "id": request_id, "ok": true,
		"consumption": consumption,
	}, true)
	_send(body)


func _error(request_id: int, reason: String, consumption: String) -> void:
	_send({
		"protocol": PROTOCOL, "id": request_id, "ok": false,
		"error": reason, "consumption": consumption,
	})


func _fatal(reason: String, consumption: String) -> void:
	_error(last_request, reason, consumption)
	_close_owner()
	exit_after_flush = 2


func _send(value: Dictionary) -> void:
	var data: PackedByteArray = (
		JSON.stringify(PhysicalArena.json_value(value)) + "\n").to_utf8_buffer()
	if data.size() > MAX_BYTES:
		_close_owner()
		quit(2)
		return
	if not outbound.is_empty():
		# Never accumulate replies behind a stalled reader.
		_close_owner()
		quit(2)
		return
	outbound = data
	outbound_started_usec = Time.get_ticks_usec()
	_flush_output()


func _flush_output() -> void:
	if outbound.is_empty():
		if exit_after_flush >= 0:
			quit(exit_after_flush)
		return
	if Time.get_ticks_usec() - outbound_started_usec >= OUTPUT_DEADLINE_USEC:
		_close_owner()
		quit(2)
		return
	var sent: Array = peer.put_partial_data(outbound)
	if int(sent[0]) != OK:
		_close_owner()
		quit(2)
		return
	outbound = outbound.slice(int(sent[1]))
	if outbound.is_empty() and exit_after_flush >= 0:
		quit(exit_after_flush)


static func _integer(value: Variant, minimum: int, maximum: int) -> bool:
	return (value is int or value is float) and is_finite(float(value)) \
		and float(value) == floorf(float(value)) \
		and float(value) >= minimum and float(value) <= maximum


func _close_owner() -> void:
	if episode != null:
		episode.close()
		episode.free()
		episode = null
