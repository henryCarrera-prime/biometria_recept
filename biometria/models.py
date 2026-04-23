import uuid

from django.db import models
from django.utils import timezone


class StoredDocument(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    original_name = models.CharField(max_length=255)
    stored_name = models.CharField(max_length=255)
    relative_path = models.CharField(max_length=1024, unique=True)
    content_type = models.CharField(max_length=255, blank=True, default="")
    size_bytes = models.BigIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "stored_documents"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.id} ({self.original_name})"


class ProcessStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    RUNNING = "running", "Running"
    PAUSED = "paused", "Paused"
    FINISHED = "finished", "Finished"
    ERROR = "error", "Error"


class ProcessExecution(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    flow_name = models.CharField(max_length=100, default="default")
    status = models.CharField(
        max_length=20, choices=ProcessStatus.choices, default=ProcessStatus.PENDING
    )
    current_step = models.CharField(max_length=100, blank=True, default="")
    next_step = models.CharField(max_length=100, blank=True, default="")
    is_finished = models.BooleanField(default=False)
    has_error = models.BooleanField(default=False)
    error_message = models.TextField(blank=True, default="")
    input_payload = models.JSONField(default=dict, blank=True)
    output_payload = models.JSONField(default=dict, blank=True)
    workspace_path = models.CharField(max_length=1024, blank=True, default="")
    document = models.ForeignKey(
        StoredDocument, null=True, blank=True, on_delete=models.SET_NULL
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        db_table = "process_executions"
        ordering = ["-created_at"]

    def mark_finished(self):
        self.status = ProcessStatus.FINISHED
        self.is_finished = True
        self.has_error = False
        self.error_message = ""
        self.finished_at = timezone.now()

    def __str__(self) -> str:
        return f"{self.id} ({self.flow_name})"


class ProcessEvent(models.Model):
    process = models.ForeignKey(
        ProcessExecution, related_name="events", on_delete=models.CASCADE
    )
    step_name = models.CharField(max_length=100)
    status = models.CharField(max_length=20)
    message = models.CharField(max_length=255, blank=True, default="")
    detail = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "process_events"
        ordering = ["created_at"]

    def __str__(self) -> str:
        return f"{self.process_id}::{self.step_name}::{self.status}"
