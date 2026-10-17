with open('/opt/molink/app/sam3_video_service.py', 'r') as f:
    content = f.read()

old_code = '''        # Add text prompt on first frame
        response = predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": text_prompt
        })

        inference_time_ms = (time.perf_counter() - start) * 1000

        # Masks are nested under response["outputs"]["out_binary_masks"]
        outputs = response.get("outputs", {})
        num_objects = len(outputs.get("out_binary_masks", []))'''

new_code = '''        # Add text prompt on first frame
        response = predictor.handle_request({
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": text_prompt
        })

        # DEBUG: Log the raw response
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"add_prompt raw response keys: {list(response.keys()) if isinstance(response, dict) else type(response)}")
        if isinstance(response, dict) and "outputs" in response:
            outputs = response["outputs"]
            logger.info(f"add_prompt outputs keys: {list(outputs.keys())}")
            logger.info(f"add_prompt out_binary_masks: {len(outputs.get('out_binary_masks', []))}")
            logger.info(f"add_prompt out_probs: {outputs.get('out_probs', [])}")
            logger.info(f"add_prompt out_obj_ids: {outputs.get('out_obj_ids', [])}")

        inference_time_ms = (time.perf_counter() - start) * 1000

        # Masks are nested under response["outputs"]["out_binary_masks"]
        outputs = response.get("outputs", {})
        num_objects = len(outputs.get("out_binary_masks", []))'''

content = content.replace(old_code, new_code)

with open('/opt/molink/app/sam3_video_service.py', 'w') as f:
    f.write(content)

print('Done')