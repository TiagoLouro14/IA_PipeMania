test1to9:
	@echo "Running tests 1 to 9"
	@for i in $$(seq 1 9); do \
		python3.8 ../proj2324base/pipe.py < test-$$(printf "%02d" $$i).txt > test-$$(printf "%02d" $$i).out; \
		diff test-$$(printf "%02d" $$i).out test-$$(printf "%02d" $$i).txt > test-$$(printf "%02d" $$i).diff; \
		if [ -s test-$$(printf "%02d" $$i).diff ]; then \
			echo "test $$i: ❌"; \
		else \
			echo "test $$i: ✅"; \
		fi; \
	done

clean:
	@echo "Cleaning up test output and diff files"
	@rm -f test-*.out test-*.diff