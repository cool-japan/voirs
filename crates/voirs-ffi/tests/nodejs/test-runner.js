#!/usr/bin/env node

/**
 * VoiRS Node.js Test Runner
 * 
 * Comprehensive test suite for VoiRS FFI Node.js bindings
 * Runs all test categories with proper reporting and error handling
 */

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');

// Test configuration
const TEST_CONFIG = {
    timeout: 30000, // 30 seconds timeout for each test suite
    parallel: true, // Run tests in parallel when possible
    verboseOutput: process.env.VERBOSE === 'true',
    skipSlowTests: process.env.VOIRS_SKIP_SLOW_TESTS === 'true' || process.env.CI === 'true'
};

// Test suites to run
const TEST_SUITES = [
    {
        name: 'Unit Tests',
        file: 'unit-tests.js',
        description: 'Core functionality unit tests',
        critical: true
    },
    {
        name: 'Integration Tests', 
        file: 'integration-tests.js',
        description: 'End-to-end integration tests',
        critical: true
    },
    {
        name: 'Error Handling Tests',
        file: 'error-tests.js', 
        description: 'Error handling and edge cases',
        critical: true
    },
    {
        name: 'Performance Tests',
        file: 'performance-tests.js',
        description: 'Performance benchmarks and stress tests',
        critical: false,
        skipInFastMode: true
    },
    {
        name: 'Memory Tests',
        file: 'memory-tests.js',
        description: 'Memory usage and leak detection',
        critical: false,
        skipInFastMode: true
    },
    {
        name: 'Concurrency Tests',
        file: 'concurrent-tests.js',
        description: 'Multi-threaded and concurrent operations',
        critical: true
    }
];

class TestRunner {
    constructor() {
        this.results = [];
        this.startTime = Date.now();
        this.failedSuites = [];
    }

    log(message, level = 'info') {
        const timestamp = new Date().toISOString();
        const prefix = {
            'info': '📝',
            'success': '✅', 
            'error': '❌',
            'warning': '⚠️',
            'debug': '🔍'
        }[level] || '📝';
        
        console.log(`[${timestamp}] ${prefix} ${message}`);
    }

    async runSuite(suite) {
        const suitePath = path.join(__dirname, suite.file);
        
        // Skip non-critical tests in fast mode
        if (TEST_CONFIG.skipSlowTests && suite.skipInFastMode) {
            this.log(`Skipping ${suite.name} (fast mode enabled)`, 'warning');
            return { 
                name: suite.name, 
                status: 'skipped', 
                duration: 0,
                output: 'Skipped in fast mode'
            };
        }

        // Check if test file exists
        if (!fs.existsSync(suitePath)) {
            this.log(`Test file not found: ${suite.file}`, 'error');
            return {
                name: suite.name,
                status: 'error',
                duration: 0,
                output: `Test file not found: ${suite.file}`
            };
        }

        this.log(`Running ${suite.name}...`, 'info');
        const startTime = Date.now();

        return new Promise((resolve) => {
            const child = spawn('node', [suitePath], {
                stdio: TEST_CONFIG.verboseOutput ? 'inherit' : 'pipe',
                env: { ...process.env, NODE_ENV: 'test' }
            });

            let output = '';
            let errorOutput = '';

            if (!TEST_CONFIG.verboseOutput) {
                child.stdout?.on('data', (data) => {
                    output += data.toString();
                });

                child.stderr?.on('data', (data) => {
                    errorOutput += data.toString();
                });
            }

            // Set timeout
            const timeout = setTimeout(() => {
                child.kill('SIGTERM');
                this.log(`${suite.name} timed out after ${TEST_CONFIG.timeout}ms`, 'error');
            }, TEST_CONFIG.timeout);

            child.on('close', (code) => {
                clearTimeout(timeout);
                const duration = Date.now() - startTime;
                const fullOutput = output + (errorOutput ? `\nSTDERR:\n${errorOutput}` : '');

                const result = {
                    name: suite.name,
                    status: code === 0 ? 'passed' : 'failed',
                    duration,
                    output: fullOutput,
                    exitCode: code
                };

                if (code === 0) {
                    this.log(`${suite.name} passed (${duration}ms)`, 'success');
                } else {
                    this.log(`${suite.name} failed with exit code ${code} (${duration}ms)`, 'error');
                    if (suite.critical) {
                        this.failedSuites.push(suite.name);
                    }
                }

                resolve(result);
            });

            child.on('error', (error) => {
                clearTimeout(timeout);
                const duration = Date.now() - startTime;
                
                this.log(`${suite.name} encountered error: ${error.message}`, 'error');
                
                resolve({
                    name: suite.name,
                    status: 'error',
                    duration,
                    output: `Error: ${error.message}`,
                    error: error.message
                });
            });
        });
    }

    async runAllTests() {
        this.log('🚀 Starting VoiRS Node.js Test Suite', 'info');
        this.log(`Configuration: timeout=${TEST_CONFIG.timeout}ms, parallel=${TEST_CONFIG.parallel}, fastMode=${TEST_CONFIG.skipSlowTests}`, 'debug');

        // Run tests in parallel or sequential based on config
        if (TEST_CONFIG.parallel) {
            const promises = TEST_SUITES.map(suite => this.runSuite(suite));
            this.results = await Promise.all(promises);
        } else {
            for (const suite of TEST_SUITES) {
                const result = await this.runSuite(suite);
                this.results.push(result);
            }
        }

        this.generateReport();
    }

    generateReport() {
        const totalDuration = Date.now() - this.startTime;
        const passed = this.results.filter(r => r.status === 'passed').length;
        const failed = this.results.filter(r => r.status === 'failed').length;
        const errors = this.results.filter(r => r.status === 'error').length;
        const skipped = this.results.filter(r => r.status === 'skipped').length;
        const total = this.results.length;

        this.log('\n📊 Test Results Summary', 'info');
        this.log('='.repeat(50), 'info');
        
        this.results.forEach(result => {
            const status = {
                'passed': '✅',
                'failed': '❌', 
                'error': '💥',
                'skipped': '⏭️'
            }[result.status];
            
            this.log(`${status} ${result.name} (${result.duration}ms)`, 'info');
            
            if (result.status === 'failed' || result.status === 'error') {
                if (TEST_CONFIG.verboseOutput && result.output) {
                    console.log('  Output:', result.output.split('\n').slice(0, 5).join('\n'));
                }
            }
        });

        this.log('='.repeat(50), 'info');
        this.log(`Total: ${total}, Passed: ${passed}, Failed: ${failed}, Errors: ${errors}, Skipped: ${skipped}`, 'info');
        this.log(`Total Duration: ${totalDuration}ms`, 'info');

        // Exit with appropriate code
        if (failed > 0 || errors > 0) {
            if (this.failedSuites.length > 0) {
                this.log(`Critical test suites failed: ${this.failedSuites.join(', ')}`, 'error');
            }
            this.log('❌ Test suite failed', 'error');
            process.exit(1);
        } else {
            this.log('✅ All tests passed successfully!', 'success');
            process.exit(0);
        }
    }
}

// Main execution
async function main() {
    try {
        // Check if we're in the right directory
        const parentDir = path.resolve(__dirname, '../..');
        const packageJsonPath = path.join(parentDir, 'package.json');
        
        if (!fs.existsSync(packageJsonPath)) {
            console.error('❌ Could not find package.json. Make sure to run tests from the correct directory.');
            process.exit(1);
        }

        const runner = new TestRunner();
        await runner.runAllTests();
    } catch (error) {
        console.error('❌ Test runner failed:', error.message);
        if (TEST_CONFIG.verboseOutput) {
            console.error(error.stack);
        }
        process.exit(1);
    }
}

// Run if this file is executed directly
if (require.main === module) {
    main();
}

module.exports = { TestRunner, TEST_CONFIG };