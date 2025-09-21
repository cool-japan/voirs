#!/usr/bin/env python3

"""
Comprehensive Cross-Language Consistency Tests for VoiRS FFI Bindings

This test suite validates that all language bindings (C, Python, Node.js) provide
consistent behavior, APIs, and results across different platforms and usage patterns.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import numpy as np
import pytest


class ConsistencyTestFramework:
    """Framework for running cross-language consistency tests"""
    
    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)
        self.results = {}
        self.temp_dir = Path(tempfile.mkdtemp())
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup temp directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def run_python_test(self, test_name: str, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a Python test and return results"""
        try:
            # Import VoiRS Python bindings
            sys.path.insert(0, str(self.test_dir.parent.parent))
            import voirs_ffi
            
            pipeline = voirs_ffi.VoirsPipeline()
            
            # Basic synthesis test
            if test_name == "basic_synthesis":
                result = pipeline.synthesize(test_params["text"])
                return {
                    "success": True,
                    "audio_length": len(result.samples),
                    "sample_rate": result.sample_rate,
                    "channels": result.channels,
                    "duration": result.duration,
                    "language": "python"
                }
            
            # Metrics test
            elif test_name == "synthesis_with_metrics":
                result = pipeline.synthesize_with_metrics(test_params["text"])
                return {
                    "success": True,
                    "audio_length": len(result.audio.samples),
                    "sample_rate": result.audio.sample_rate,
                    "duration": result.audio.duration,
                    "processing_time_ms": result.metrics.processing_time_ms,
                    "real_time_factor": result.metrics.real_time_factor,
                    "language": "python"
                }
            
            # Voice listing test
            elif test_name == "list_voices":
                voices = pipeline.list_voices()
                return {
                    "success": True,
                    "voice_count": len(voices),
                    "voices": [{"id": v.id, "name": v.name, "language": v.language} for v in voices],
                    "language": "python"
                }
            
            # Batch synthesis test
            elif test_name == "batch_synthesis":
                results = pipeline.batch_synthesize(test_params["texts"])
                return {
                    "success": True,
                    "batch_size": len(results),
                    "total_duration": sum(r.audio.duration for r in results),
                    "avg_real_time_factor": sum(r.metrics.real_time_factor for r in results) / len(results),
                    "language": "python"
                }
            
            # Audio analysis test
            elif test_name == "audio_analysis":
                audio_result = pipeline.synthesize(test_params["text"])
                analysis = pipeline.analyze_audio(audio_result)
                return {
                    "success": True,
                    "rms_energy": analysis.rms_energy,
                    "zero_crossing_rate": analysis.zero_crossing_rate,
                    "spectral_centroid": analysis.spectral_centroid,
                    "language": "python"
                }
            
            else:
                return {"success": False, "error": f"Unknown test: {test_name}", "language": "python"}
                
        except Exception as e:
            return {"success": False, "error": str(e), "language": "python"}
    
    def run_nodejs_test(self, test_name: str, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a Node.js test and return results"""
        try:
            # Create a temporary Node.js test script
            test_script = self.temp_dir / f"test_{test_name}.js"
            
            # Generate Node.js test code
            nodejs_code = self._generate_nodejs_test_code(test_name, test_params)
            
            with open(test_script, 'w') as f:
                f.write(nodejs_code)
            
            # Run Node.js test
            result = subprocess.run(
                ['node', str(test_script)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(self.test_dir.parent.parent)
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "language": "nodejs"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e), "language": "nodejs"}
    
    def _generate_nodejs_test_code(self, test_name: str, test_params: Dict[str, Any]) -> str:
        """Generate Node.js test code for a specific test"""
        
        if test_name == "basic_synthesis":
            return f"""
const {{ VoirsPipeline }} = require('./index.js');

async function test() {{
    try {{
        const pipeline = new VoirsPipeline();
        const result = await pipeline.synthesize('{test_params["text"]}');
        
        console.log(JSON.stringify({{
            success: true,
            audio_length: result.samples.length,
            sample_rate: result.sampleRate,
            channels: result.channels,
            duration: result.duration,
            language: "nodejs"
        }}));
    }} catch (error) {{
        console.log(JSON.stringify({{
            success: false,
            error: error.message,
            language: "nodejs"
        }}));
    }}
}}

test();
"""
        
        elif test_name == "synthesis_with_metrics":
            return f"""
const {{ VoirsPipeline }} = require('./index.js');

async function test() {{
    try {{
        const pipeline = new VoirsPipeline();
        const result = await pipeline.synthesizeWithMetrics('{test_params["text"]}');
        
        console.log(JSON.stringify({{
            success: true,
            audio_length: result.audio.samples.length,
            sample_rate: result.audio.sampleRate,
            duration: result.audio.duration,
            processing_time_ms: result.metrics.processingTimeMs,
            real_time_factor: result.metrics.realTimeFactor,
            language: "nodejs"
        }}));
    }} catch (error) {{
        console.log(JSON.stringify({{
            success: false,
            error: error.message,
            language: "nodejs"
        }}));
    }}
}}

test();
"""
        
        elif test_name == "list_voices":
            return f"""
const {{ VoirsPipeline }} = require('./index.js');

async function test() {{
    try {{
        const pipeline = new VoirsPipeline();
        const voices = await pipeline.listVoices();
        
        console.log(JSON.stringify({{
            success: true,
            voice_count: voices.length,
            voices: voices.map(v => ({{ id: v.id, name: v.name, language: v.language }})),
            language: "nodejs"
        }}));
    }} catch (error) {{
        console.log(JSON.stringify({{
            success: false,
            error: error.message,
            language: "nodejs"
        }}));
    }}
}}

test();
"""
        
        elif test_name == "batch_synthesis":
            texts = json.dumps(test_params["texts"])
            return f"""
const {{ VoirsPipeline }} = require('./index.js');

async function test() {{
    try {{
        const pipeline = new VoirsPipeline();
        const results = await pipeline.batchSynthesize({texts});
        
        const totalDuration = results.reduce((sum, r) => sum + r.audio.duration, 0);
        const avgRealTimeFactor = results.reduce((sum, r) => sum + r.metrics.realTimeFactor, 0) / results.length;
        
        console.log(JSON.stringify({{
            success: true,
            batch_size: results.length,
            total_duration: totalDuration,
            avg_real_time_factor: avgRealTimeFactor,
            language: "nodejs"
        }}));
    }} catch (error) {{
        console.log(JSON.stringify({{
            success: false,
            error: error.message,
            language: "nodejs"
        }}));
    }}
}}

test();
"""
        
        elif test_name == "audio_analysis":
            return f"""
const {{ VoirsPipeline }} = require('./index.js');

async function test() {{
    try {{
        const pipeline = new VoirsPipeline();
        const audioResult = await pipeline.synthesize('{test_params["text"]}');
        const analysis = await pipeline.analyzeAudio(audioResult);
        
        console.log(JSON.stringify({{
            success: true,
            rms_energy: analysis.rmsEnergy,
            zero_crossing_rate: analysis.zeroCrossingRate,
            spectral_centroid: analysis.spectralCentroid,
            language: "nodejs"
        }}));
    }} catch (error) {{
        console.log(JSON.stringify({{
            success: false,
            error: error.message,
            language: "nodejs"
        }}));
    }}
}}

test();
"""
        
        else:
            return f"""
console.log(JSON.stringify({{
    success: false,
    error: "Unknown test: {test_name}",
    language: "nodejs"
}}));
"""
    
    def run_c_test(self, test_name: str, test_params: Dict[str, Any]) -> Dict[str, Any]:
        """Run a C test and return results"""
        try:
            # Create a temporary C test program
            test_program = self.temp_dir / f"test_{test_name}.c"
            executable = self.temp_dir / f"test_{test_name}"
            
            # Generate C test code
            c_code = self._generate_c_test_code(test_name, test_params)
            
            with open(test_program, 'w') as f:
                f.write(c_code)
            
            # Compile C test
            compile_result = subprocess.run([
                'gcc', '-o', str(executable), str(test_program),
                '-I', str(self.test_dir.parent.parent / 'src'),
                '-L', str(self.test_dir.parent.parent / 'target/release'),
                '-lvoirs_ffi'
            ], capture_output=True, text=True)
            
            if compile_result.returncode != 0:
                return {
                    "success": False,
                    "error": f"Compilation failed: {compile_result.stderr}",
                    "language": "c"
                }
            
            # Run C test
            result = subprocess.run(
                [str(executable)],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {
                    "success": False,
                    "error": result.stderr,
                    "language": "c"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e), "language": "c"}
    
    def _generate_c_test_code(self, test_name: str, test_params: Dict[str, Any]) -> str:
        """Generate C test code for a specific test"""
        
        if test_name == "basic_synthesis":
            return f"""
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "voirs_ffi.h"

int main() {{
    VoirsPipeline* pipeline = voirs_pipeline_new(NULL);
    if (!pipeline) {{
        printf("{{\\"success\\": false, \\"error\\": \\"Failed to create pipeline\\", \\"language\\": \\"c\\"}}");
        return 1;
    }}
    
    AudioBuffer* result = voirs_pipeline_synthesize(pipeline, "{test_params["text"]}");
    if (!result) {{
        printf("{{\\"success\\": false, \\"error\\": \\"Synthesis failed\\", \\"language\\": \\"c\\"}}");
        voirs_pipeline_free(pipeline);
        return 1;
    }}
    
    printf("{{\\"success\\": true, \\"audio_length\\": %zu, \\"sample_rate\\": %u, \\"channels\\": %u, \\"duration\\": %f, \\"language\\": \\"c\\"}}",
           voirs_audio_buffer_length(result),
           voirs_audio_buffer_sample_rate(result),
           voirs_audio_buffer_channels(result),
           voirs_audio_buffer_duration(result));
    
    voirs_audio_buffer_free(result);
    voirs_pipeline_free(pipeline);
    return 0;
}}
"""
        
        elif test_name == "list_voices":
            return f"""
#include <stdio.h>
#include <stdlib.h>
#include "voirs_ffi.h"

int main() {{
    VoirsPipeline* pipeline = voirs_pipeline_new(NULL);
    if (!pipeline) {{
        printf("{{\\"success\\": false, \\"error\\": \\"Failed to create pipeline\\", \\"language\\": \\"c\\"}}");
        return 1;
    }}
    
    VoiceInfo* voices;
    size_t count = voirs_pipeline_list_voices(pipeline, &voices);
    
    printf("{{\\"success\\": true, \\"voice_count\\": %zu, \\"voices\\": [", count);
    for (size_t i = 0; i < count; i++) {{
        printf("{{\\"id\\": \\"%s\\", \\"name\\": \\"%s\\", \\"language\\": \\"%s\\"}}", 
               voices[i].id, voices[i].name, voices[i].language);
        if (i < count - 1) printf(",");
    }}
    printf("], \\"language\\": \\"c\\"}}}");
    
    voirs_voices_free(voices, count);
    voirs_pipeline_free(pipeline);
    return 0;
}}
"""
        
        else:
            return f"""
#include <stdio.h>

int main() {{
    printf("{{\\"success\\": false, \\"error\\": \\"Unknown test: {test_name}\\", \\"language\\": \\"c\\"}}");
    return 1;
}}
"""
    
    def compare_results(self, results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compare results across different language bindings"""
        
        comparison = {
            "consistent": True,
            "differences": [],
            "summary": {}
        }
        
        # Get all successful results
        successful_results = {lang: result for lang, result in results.items() if result.get("success", False)}
        
        if len(successful_results) < 2:
            comparison["consistent"] = False
            comparison["differences"].append("Less than 2 successful results to compare")
            return comparison
        
        # Compare common fields
        common_fields = set()
        for result in successful_results.values():
            common_fields.update(result.keys())
        
        # Remove language-specific fields
        common_fields.discard("language")
        common_fields.discard("success")
        
        for field in common_fields:
            values = {}
            for lang, result in successful_results.items():
                if field in result:
                    values[lang] = result[field]
            
            if len(set(str(v) for v in values.values())) > 1:
                comparison["consistent"] = False
                comparison["differences"].append({
                    "field": field,
                    "values": values
                })
        
        # Summary statistics
        comparison["summary"] = {
            "total_languages": len(results),
            "successful_languages": len(successful_results),
            "failed_languages": len(results) - len(successful_results),
            "consistency_score": len(successful_results) / len(results) if results else 0
        }
        
        return comparison
    
    def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete cross-language consistency test suite"""
        
        test_cases = [
            ("basic_synthesis", {"text": "Hello, this is a test of the VoiRS speech synthesis system."}),
            ("synthesis_with_metrics", {"text": "Testing metrics collection during synthesis."}),
            ("list_voices", {}),
            ("batch_synthesis", {"texts": ["First text", "Second text", "Third text"]}),
            ("audio_analysis", {"text": "Analyzing this audio for quality metrics."}),
        ]
        
        results = {}
        
        for test_name, test_params in test_cases:
            print(f"Running test: {test_name}")
            
            test_results = {}
            
            # Run Python test
            print(f"  Running Python test...")
            test_results["python"] = self.run_python_test(test_name, test_params)
            
            # Run Node.js test
            print(f"  Running Node.js test...")
            test_results["nodejs"] = self.run_nodejs_test(test_name, test_params)
            
            # Run C test (if available)
            print(f"  Running C test...")
            test_results["c"] = self.run_c_test(test_name, test_params)
            
            # Compare results
            comparison = self.compare_results(test_results)
            
            results[test_name] = {
                "test_results": test_results,
                "comparison": comparison
            }
            
            print(f"  Consistency: {'✓' if comparison['consistent'] else '✗'}")
            if not comparison["consistent"]:
                print(f"  Differences: {len(comparison['differences'])}")
        
        return results


def main():
    """Main test runner"""
    test_dir = Path(__file__).parent
    
    print("🔍 VoiRS Cross-Language Consistency Test Suite")
    print("=" * 50)
    
    with ConsistencyTestFramework(str(test_dir)) as framework:
        results = framework.run_comprehensive_test_suite()
        
        # Generate summary report
        print("\n📊 Test Results Summary")
        print("=" * 50)
        
        total_tests = len(results)
        consistent_tests = sum(1 for r in results.values() if r["comparison"]["consistent"])
        
        print(f"Total tests: {total_tests}")
        print(f"Consistent tests: {consistent_tests}")
        print(f"Inconsistent tests: {total_tests - consistent_tests}")
        print(f"Consistency rate: {consistent_tests / total_tests * 100:.1f}%")
        
        # Detailed results
        for test_name, result in results.items():
            comparison = result["comparison"]
            print(f"\n{test_name}:")
            print(f"  Consistent: {'✓' if comparison['consistent'] else '✗'}")
            print(f"  Success rate: {comparison['summary']['consistency_score']:.1f}")
            
            if comparison["differences"]:
                print(f"  Differences:")
                for diff in comparison["differences"]:
                    if isinstance(diff, dict):
                        print(f"    {diff['field']}: {diff['values']}")
                    else:
                        print(f"    {diff}")
        
        # Save detailed results
        output_file = test_dir / "consistency_test_results.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nDetailed results saved to: {output_file}")
        
        # Exit with appropriate code
        if consistent_tests == total_tests:
            print("\n✅ All tests passed! Cross-language consistency verified.")
            sys.exit(0)
        else:
            print(f"\n❌ {total_tests - consistent_tests} tests failed. Cross-language inconsistencies detected.")
            sys.exit(1)


if __name__ == "__main__":
    main()