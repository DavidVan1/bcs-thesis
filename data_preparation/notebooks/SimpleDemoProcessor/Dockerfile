FROM ubuntu

MAINTAINER CGI


RUN ls /
RUN ls /
RUN ls /

# Prepare processor script
RUN mkdir -p /home/worker/processor
COPY * /home/worker/processor/
ENTRYPOINT ["/home/worker/processor/workflow.sh"]
